# Hybrid Quantum–Classical CNNs on CIFAR-100

This repository contains deterministic experiments on CIFAR-100 with a ResNet-34 backbone, comparing:

1. **Baseline** – standard ResNet-34 (deterministic training setup).  
2. **Quantum Feature Maps (5-channel hybrid)** – quantum-inspired descriptors precomputed per image and injected as additional channels.  
3. **Quantum Head (last-layer fusion)** – a variational quantum circuit on top of the backbone features, fused with the classical logits via a learnable scalar gate.

All experiments reuse the same train/validation split, data transforms, optimizer and learning-rate schedule to enable a fair comparison.


## 1. Methods

### 1.1 Baseline: Deterministic ResNet-34

- Backbone: `torchvision.models.resnet34`, ImageNet initialization.  
- Input: 3-channel RGB, standard CIFAR-style augmentations (RandomResizedCrop, RandAugment, RandomHorizontalFlip).  
- Optimizer: `AdamW` with `OneCycleLR`.  
- Loss: Cross-entropy with label smoothing.  
- All seeds and CUDA flags set for deterministic behaviour.

> Baseline training script is compatible with the same split file:
> `runs/cifar100_baseline_det/split_idx_cifar100.npz`.

---

### 1.2 Quantum Feature Maps (5-Channel Hybrid)

**Script:** `cifar100_quantum_channels.py`

Idea:

- For each image, downsample to extract a small vector of N_QUBITS scalars.
- Feed this vector into two small circuits:
  - `qnode_edge`: ring-entangled circuit intended to capture edge-like correlations.
  - `qnode_tex`: star-entangled circuit with additional phase interactions for texture-like correlations.
- Each circuit outputs an N_QUBITS-dimensional descriptor.  
- The descriptors are mapped to small 2D grids and resized to the input resolution, yielding two additional channels:
  - `RGB + QE + QT → 5 × H × W`.
- The first ResNet convolution (`conv1`) is expanded from 3 to 5 input channels.

The quantum descriptors are precomputed once and cached to:

- `train_q_edge*.npz`, `train_q_tex*.npz`  
- `val_q_edge*.npz`, `val_q_tex*.npz`  
- `test_q_edge*.npz`, `test_q_tex*.npz`

---

### 1.3 Quantum Head with Learnable Fusion (Last-Layer Hybrid)

**Script:** `cifar100_hybrid_Qhead.py`

Idea:

- Keep the ResNet-34 backbone and classical classifier identical to the baseline (same number of features, same final linear layer).
- Add a parallel quantum branch on top of the backbone features: 
  1. Variational quantum circuit (`AngleEmbedding` + `StronglyEntanglingLayers`) - with learnble weights
  2. Linear mapping back to `num_classes`.

- Fuse the classical and quantum logits via a learnable scalar gate:
  
  logits = logits_cls + α · logits_q,   α = sigmoid(α_raw) ∈ (0, 1)
  
Training is performed in two phases:

Phase A (warm-up):

α initialized so that α ≈ 0 (quantum branch effectively off).

Quantum parameters frozen; backbone and classical head train as in the baseline.

Phase B (fusion):

α becomes learnable; quantum branch and backbone are both trainable.

α is monitored once per epoch to see how strong the quantum contribution becomes.



## 2. Results 

Baseline ResNet-34	~0.829	Deterministic baseline

Hybrid 5-channel (quantum features)	~0.829–0.83	Very close to baseline

Hybrid last-layer quantum head	~0.8279	

So far, the main observation is that small quantum or quantum-inspired modules can be integrated without destroying baseline performance, 
but also do not trivially yield a large accuracy improvement under this controlled setup.


## 3. Environment

conda create -n qml-cifar python=3.11

conda activate qml-cifar

pip install torch torchvision torchaudio

pip install pennylane pennylane-lightning

pip install tqdm pillow

DLfunctions.py (containing train_model) is available in the Python path.


## 4. Notes and Extensions

Both hybrid variants are deliberately close to the baseline in terms of parameter count and training setup, to isolate the effect of the quantum components.

The current implementation uses simulation on classical hardware.

The circuits and dataflow are structured to be portable to real quantum backends, subject to device constraints.

