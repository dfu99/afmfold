import os
import math
import copy
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from biotite.structure.io.pdb import PDBFile
from biotite.structure import filter_solvent, get_chains, AtomArray
import tempfile
import mdtraj as md

from afmfold.cnn import CNSteerableCNN
from afmfold.utils import move_all_tensors_in_device

def guidance_scale(t, func_type, y_max=1.0, alpha=5, beta=10, gamma=0.5, t_start=0.0, t_end=1.0, **kwargs):
    assert func_type in ["linear", "ease_in", "ease_out", "sigmoid", "constant"], "Invalid function type"
    # To numpy.
    if isinstance(t, torch.Tensor):
        t_np = t.detach().cpu().numpy()
    elif isinstance(t, float):
        t_np = np.array(t)
    elif isinstance(t, np.ndarray):
        t_np = t
    else:
        raise TypeError("Input must be torch.Tensor, float, or np.ndarray")

    # relative time when t_start = 0 and t_end = 1.0.
    if t_np >= t_start:
        t_local = (t_np - t_start) / (t_end - t_start) * 1.0
    else:
        t_local = 0.0
        
    # Compute with NumPy.
    if func_type == "linear":
        result = y_max * t_local

    elif func_type == "ease_in":
        result = y_max * (np.exp(alpha * t_local) - 1) / (np.exp(alpha) - 1)

    elif func_type == "ease_out":
        result = y_max * (1 - (np.exp(-alpha * t_local) - np.exp(-alpha)) / (1 - np.exp(-alpha)))

    elif func_type == "sigmoid":
        s = 1 / (1 + np.exp(-beta * (t_local - gamma)))
        s_min = 1 / (1 + np.exp(-beta * (-gamma)))
        s_max = 1 / (1 + np.exp(-beta * (gamma)))
        result = y_max * (s - s_min) / (s_max - s_min)
    
    elif func_type == "constant":
        result = y_max * np.ones_like(t_local)

    # Return by float.
    return float(result)
    
class DomainDistanceLoss:
    def __init__(self, model_path, image_path, domain_pairs, manual=None, model=None, in_nm=False, device="cuda"):
        self.model_path = model_path
        if os.path.exists(model_path) and model is None:
            model, _, _ = CNSteerableCNN.load_from_checkpoint(model_path)
            self.model = model.to(device)
            self.model.eval()
        elif model is None:
            self.model = model
        else:
            self.model = model.to(device)
            self.model.eval()
        self.image_path = image_path
        
        # Set up the domain pair.
        self.domain_indices_list = domain_pairs
        self.Npairs = len(self.domain_indices_list)
        
        self.manual = manual
        self.in_nm = in_nm
        self.device = device
        
        # Load image and predict the domain distance.
        if os.path.exists(image_path):
            self.image = self.load_image(image_path)
        else:
            self.image = None
            
        if self.manual is None or len(self.manual) == 0:
            if self.image is None:
                self.target = None
            else:
                self.target = self.compute_target().detach().cpu()
        else:
            if isinstance(self.manual, list) or isinstance(self.manual, tuple):
                self.target = torch.tensor(self.manual)
            elif isinstance(self.manual, np.ndarray):
                self.target = torch.from_numpy(self.manual)
            else:
                self.target = self.manual
        
        if self.in_nm:
            self.target *= 10.0
        
    def load_image(self, image_path):
        if image_path.endswith(".npy"):
            image = torch.from_numpy(np.load(image_path))
        elif image_path.endswith(".pt"):
            image = torch.load(image_path)
        else:
            raise ValueError(f"Invalid file type: {image_path}.")
        H, W = image.shape[-2:]
        image = image.reshape((-1, H, W))
        return image

    def compute_target(self):
        image = move_all_tensors_in_device(self.image, device=self.device)[0]
        image = image.unsqueeze(1).to(torch.float)
        
        output = self.model(image)  # [B, D] tensor.
        target = torch.mean(output, dim=0)
        return target
        
    def compute_loss(self, x_coords, atom_array):
        # If no target is set, return zero loss.
        if self.target is None:
            return 0.0 * torch.sum(x_coords)
        
        # Check the shape
        B, N, _ = x_coords.shape
        device = x_coords.device
        
        # Convert (N_atoms,) boolean array → torch.Tensor
        is_calpha = torch.from_numpy((atom_array.atom_name == "CA")).to(device)
        ca_indices = torch.nonzero(is_calpha, as_tuple=False).squeeze(dim=1)  # shape: [N_calpha]

        # Reshape ca_indices to [1, N_calpha, 1] and repeat B times → [B, N_calpha, 1]
        ca_indices_expand = ca_indices.view(1, -1, 1).expand(B, -1, 1)

        # Extract Cα atom coordinates along dim=1 using gather
        ca_coords = torch.gather(x_coords, dim=1, index=ca_indices_expand.expand(-1, -1, 3))  # shape: [B, N_calpha, 3]

        # Compute whether each Cα atom belongs to a domain
        is_domain1, is_domain2 = self.assign_domain_mask(len(ca_indices), device=device)  # [Npairs, N_calpha]
        
        # Broadcast: [B, Npairs, N_calpha, 3]
        ca_coords_exp = ca_coords.unsqueeze(1).expand(B, self.Npairs, -1, 3)
        is_domain1 = is_domain1.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 3)
        is_domain2 = is_domain2.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 3)

        # Mask coordinates for each domain and compute centroids
        domain1_coords = ca_coords_exp * is_domain1
        domain2_coords = ca_coords_exp * is_domain2

        domain1_sum = domain1_coords.sum(dim=2)  # [B, Npairs, 3]
        domain2_sum = domain2_coords.sum(dim=2)
        
        domain1_count = is_domain1[:, :, :, 0].sum(dim=2).clamp(min=1).unsqueeze(-1)
        domain2_count = is_domain2[:, :, :, 0].sum(dim=2).clamp(min=1).unsqueeze(-1)

        domain1_com = domain1_sum / domain1_count  # [B, Npairs, 3]
        domain2_com = domain2_sum / domain2_count

        # Compute distance
        dist = torch.norm(domain1_com - domain2_com, dim=-1)  # [B, Npairs]

        # Compute loss (squared difference with the target)
        target = self.target.to(device).view(1, self.Npairs)  # [1, Npairs]
        sq_diff = (dist - target) ** 2  # [B, Npairs]
        
        loss = torch.sum(sq_diff)  # scalar
        
        return loss
    
    def assign_domain_mask(self, N, device=None):
        # Create CA atom masks for each domain pair: is_domain1, is_domain2 with shape [Npairs, N_calpha]
        if device is None:
            device = self.device
        domain1_masks = []
        domain2_masks = []
        for d1, d2 in self.domain_indices_list:
            d1_mask = torch.zeros(N, dtype=torch.bool)
            d2_mask = torch.zeros(N, dtype=torch.bool)
            d1_mask[torch.from_numpy(d1)] = True
            d2_mask[torch.from_numpy(d2)] = True
            domain1_masks.append(d1_mask)
            domain2_masks.append(d2_mask)
            
        # [Npairs, N_calpha]
        is_domain1 = torch.stack(domain1_masks).to(device)
        is_domain2 = torch.stack(domain2_masks).to(device)
        
        return is_domain1, is_domain2
