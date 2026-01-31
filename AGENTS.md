# AFM-Fold memory notes (local)

## Paper methodology (high level)
- Goal: infer 3D conformation from a single AFM image.
- Core idea: predict low-dimensional CVs (inter-domain distances) from AFM image with a rotation-equivariant CNN, then guide AF3/Protenix diffusion to generate a 3D structure that matches those CVs.

## Training data preparation (paper)
- Generate candidate conformations by guiding AF3/Protenix with CV targets placed on a grid around a reference structure.
  - CV grid range: each CV in [min(0.0 nm, CV_ref - 5.0 nm), CV_ref + 5.0 nm], step 0.5 nm.
- Geometric sanitization with MolProbity thresholds:
  - MolProbity score <= 4.0
  - Clash score <= 70.0
  - Ramachandran favored >= 90%
  - Rotamer outlier <= 50%
- Render pseudo-AFM images from the sanitized conformations.
- Train g-CNN to regress CVs from pseudo-AFM images (MSE loss).

## Inference (paper)
- For each AFM image:
  1) g-CNN predicts CVs (inter-domain distances).
  2) AF3/Protenix sampling is guided by those CVs to generate a structure.
- Rigid-body fitting against the AFM image is used for evaluation (correlation coefficient), not as the main estimator.

## Important clarifications
- This is not a nearest-neighbor lookup against a precomputed conformer library.
- The CNN does not directly output a 3D structure; it outputs CVs used to guide structure generation.
- Time-series HS-AFM can be processed frame-by-frame to yield an atomistic-resolution *sequence*, but frames are inferred independently unless extra temporal constraints are added.

## Integrin caveat (user note)
- Large-scale conformational changes (e.g., BC -> EC/EO) may not be reachable by CV-guided Protenix from a single reference.
- Consider multiple reference structures, different CVs, or an external ensemble (MD, AlphaFlow, BioEmu, AF2 MSA subsampling) to cover the conformational space before CNN training.
