# Convolutional Restricted Hopfield Network (CRHN)

A PyTorch implementation of Convolutional Restricted Hopfield Networks (CRHNs) for robust auto-associative memory retrieval and pattern reconstruction under noisy, corrupted, and adversarial conditions.

The project combines:

- Convolutional feature extraction
- Restricted Hopfield Network (RHN) dynamics
- Attractor-based memory retrieval
- Gradient-free Subspace Rotation Algorithm (SRA) training

to build a robust associative memory framework for high-dimensional data such as images.

-------------------------------------------------------------------------------

# Overview

Traditional associative memory models such as classical Hopfield Networks suffer from:

- limited memory capacity,
- poor scalability to high-dimensional images,
- weak robustness under adversarial perturbations.

This repository implements a Convolutional Restricted Hopfield Network (CRHN) that addresses these limitations by integrating:

1. A convolutional encoder-decoder architecture
2. A Restricted Hopfield Network operating in latent space
3. Iterative attractor dynamics for memory retrieval
4. Orthogonality-preserving subspace rotation training

The model progressively refines corrupted inputs until convergence to a stored memory pattern.

-------------------------------------------------------------------------------

# Features

- Robust associative memory retrieval
- Iterative fixed-point reconstruction
- Convolutional latent-space memory
- Gradient-free RHN training
- Subspace Rotation Algorithm (SRA)
- Resistance to:
  - Gaussian noise
  - Occlusion
  - Brightness shift
  - Contrast variation
  - Adversarial perturbations
- PyTorch implementation

-------------------------------------------------------------------------------

# Convolutional Restricted Hopfield Network

The CRHN architecture consists of:

Input Image
     ↓
Convolutional Encoder
     ↓
Latent Representation
     ↓
Restricted Hopfield Network
     ↓
Decoder Reconstruction
     ↓
Retrieved Pattern

The recurrent retrieval process iteratively projects corrupted patterns toward stable attractors stored in latent space.

-------------------------------------------------------------------------------

# Restricted Hopfield Network (RHN)

The RHN implementation uses:

- Orthogonal weight initialization
- Iterative attractor dynamics
- Left/Right Subspace Rotation updates
- Gradient-free learning

Unlike conventional backpropagation-based associative memories, the RHN is trained using a Subspace Rotation Algorithm (SRA) that preserves geometric structures in latent space.

Key properties:

- Improved robustness
- Better convergence stability
- Enhanced memory capacity
- Strong adversarial resistance

-------------------------------------------------------------------------------

# Subspace Rotation Algorithm (SRA)

The Subspace Rotation Algorithm updates RHN weights using SVD-based orthogonal rotations instead of gradient descent.

Advantages:

- Avoids gradient instability
- Preserves orthogonality
- Stabilizes attractor dynamics
- Improves retrieval robustness

The training alternates between:

- Left subspace rotation
- Right subspace rotation

to progressively refine associative memory representations.

-------------------------------------------------------------------------------

# Example Usage

## Import Model

from crhn import Con_RHN

## Initialize Model

model = Con_RHN(
    kernel_sizes=[3,3,3],
    num_filters=[3,16,32],
    fc_dims=[512,128],
    rhn_arch=[128,100],
    input_size=32,
    device='cuda'
)

## Train

model.train_all_model(images, epochs=100, lr=1e-4)

## Query / Retrieve Memory

reconstructed = model.query(corrupted_images)

-------------------------------------------------------------------------------

# Iterative Retrieval

The retrieval process repeatedly performs:

1. Convolutional encoding
2. RHN attractor update
3. Decoder reconstruction

until convergence:

for iteration in range(max_iterations):
    x = model.forward(x)

This iterative refinement mimics hippocampal pattern completion dynamics.

-------------------------------------------------------------------------------

# Experimental Results

The proposed CRHN demonstrates strong robustness compared to:

- Modern Hopfield Networks (MHNs)
- Predictive Coding Networks (PCNs)

under:

- adversarial attacks
- severe corruption
- noisy inputs
- brightness changes
- contrast distortions

Experiments on the STL dataset show substantially lower reconstruction error and improved retrieval stability.

-------------------------------------------------------------------------------

# Related Work

This implementation is related to several associative memory frameworks:

- Classical Hopfield Networks
- Modern Hopfield Networks
- Dense Associative Memories
- Predictive Coding Networks
- Feature-space associative memories
- Continuous attractor models

-------------------------------------------------------------------------------

# Citation

If you use this repository in your research, please cite:

@article{lin2026crhn,
  title={Robust Auto-associative Memory via Convolutional Restricted Hopfield Networks},
  author={Lin, Ci and Yeap, Tet and Kiringa, Iluju},
  journal={Under Review},
  year={2026}
}

-------------------------------------------------------------------------------

# Future Work

Potential future extensions include:

- Bidirectional associative memory
- Transformer-integrated RHNs
- Sparse associative retrieval
- Online continual memory
- Multimodal associative memory
- Feature-space retrieval
- Large-scale memory systems

-------------------------------------------------------------------------------

# License

This project is released under the MIT License.

-------------------------------------------------------------------------------

# Acknowledgement

This work was developed at the University of Ottawa, School of Electrical Engineering and Computer Science.

The project is inspired by:

- biological hippocampal memory systems
- attractor neural dynamics
- modern associative memory theory
- predictive coding frameworks
- transformer attention mechanisms
