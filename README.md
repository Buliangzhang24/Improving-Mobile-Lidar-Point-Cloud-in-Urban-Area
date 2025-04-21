# Point Cloud Processing Pipeline
This repository contains code for processing point cloud data as part of a thesis project. The workflow follows these main steps: preprocessing, registration, denoising, evaluation, and reconstruction.

## Prerequisites
First create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate pointcloud-env
```
## Code Structure
### 1. Preprocessing
a00Pre-Progress_From_las_to_ply.py: Converts LAS point cloud files to PLY format for easier processing.

### 2. Registration
a01Registration_Align.py: Performs initial alignment of MLS (Mobile Laser Scanning) and TLS (Terrestrial Laser Scanning) point clouds.

a01Registration_ICP.py: Refines the alignment using Iterative Closest Point (ICP) algorithm.

### 3. Denoising
`02Denoise_PointNet.py`: Denoising using PointNet architecture

a02Denoise_CNN.py: Denoising using CNN-based approach

a02Denoise_KNN.py: Denoising using K-nearest neighbors filtering

a02Denoise_Manifold.py: Denoising using manifold learning

a02Denoise_Normal.py: Normal-based denoising

a02Denoise_PointFilter.py: Point filtering techniques

a02Denoise_Random_Sampling.py: Random sampling for noise reduction

### 4. Evaluation
a03Evaluation_Score.py: Calculates evaluation metrics for denoising quality

b02EvaluationModel_HD.py: Computes Hausdorff Distance for surface comparison

### 5. Reconstruction
b013DReconstruction_Delaunay.py: Performs 3D surface reconstruction using Delaunay triangulation

## Workflow Order
Start with preprocessing (1_preprocessing)

Perform registration (2_registration)

Apply denoising methods (3_denoising)

Evaluate results (4_evaluation)

Reconstruct surfaces (5_reconstruction)

For best results, run the scripts in this order following the workflow diagram. Each step depends on the outputs from previous steps.
