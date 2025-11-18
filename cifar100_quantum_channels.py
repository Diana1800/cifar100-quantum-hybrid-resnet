import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import RandAugment

import pennylane as qml

from DLfunctions import train_model

# -------------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

SEED         = 42
SPLIT_SEED   = 42
DATA_DIR     = "./data"
RUN_DIR      = Path("./runs/cifar100_quantum_hybrid_det")
SPLIT_PATH   = Path("./runs/cifar100_baseline_det/split_idx_cifar100.npz")

MODEL_NAME   = "resnet34"
PRETRAINED   = True
IMG_SIZE     = 224
EPOCHS       = 40
BATCH_SIZE   = 128
NUM_WORKERS  = 0   # keep 0 for strict determinism

LR             = 3e-4
WEIGHT_DECAY   = 1e-4
MAX_LR_ONECYCLE = 7e-4
LABEL_SMOOTH   = 0.05

# Quantum feature map configuration
N_QUBITS           = 4
QUANTUM_DEPTH      = 1
QUANTUM_IMAGE_SIZE = 28
PRECOMPUTE_QFEATS  = True
FORCE_RECOMPUTE    = False

TRAIN_Q_EDGE = RUN_DIR / "train_q_edge5.npz"
TRAIN_Q_TEX  = RUN_DIR / "train_q_tex5.npz"
VAL_Q_EDGE   = RUN_DIR / "val_q_edge5.npz"
VAL_Q_TEX    = RUN_DIR / "val_q_tex5.npz"
TEST_Q_EDGE  = RUN_DIR / "test_q_edge5.npz"
TEST_Q_TEX   = RUN_DIR / "test_q_tex5.npz"

RUN_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_all_seeds(SEED)

# -------------------------------------------------------------------------
# Transforms (kept identical to baseline experiment)
# -------------------------------------------------------------------------
spatial_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=7),
])

spatial_val = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
])

to_tensor = transforms.ToTensor()

# Normalize 5 channels: RGB + (QE, QT) left as mean=0,std=1
norm5 = transforms.Normalize(
    mean=[0.485, 0.456, 0.406, 0.0, 0.0],
    std=[0.229, 0.224, 0.225, 1.0, 1.0],
)

random_erasing = transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3))

# -------------------------------------------------------------------------
# CIFAR-100 and train/val split (reuse baseline split)
# -------------------------------------------------------------------------
train_raw = datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
test_raw  = datasets.CIFAR100(root=DATA_DIR, train=False, download=True)

CLASS_NAMES = train_raw.classes
NUM_CLASSES = 100

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Baseline split not found at {SPLIT_PATH}. Run the baseline script once."
    )

split_dict = np.load(SPLIT_PATH)
train_idx = split_dict["train_idx"].tolist()
val_idx   = split_dict["val_idx"].tolist()

print(f"Loaded split from {SPLIT_PATH}")
print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_raw)}")

# -------------------------------------------------------------------------
# PennyLane devices & circuits
#   Two small circuits:
#   - qnode_edge: edge-like descriptor
#   - qnode_tex : texture-like descriptor
# -------------------------------------------------------------------------
def make_device(n_wires: int):
    """
    Try GPU-accelerated simulators first, then fall back to default.qubit.
    """
    for backend in ["lightning.gpu", "lightning.qubit", "default.qubit"]:
        try:
            return qml.device(backend, wires=n_wires)
        except Exception:
            continue
    # Fallback 
    return qml.device("default.qubit", wires=n_wires)


dev_edge = make_device(N_QUBITS)
dev_tex  = make_device(N_QUBITS)


@qml.qnode(dev_edge, interface="autograd")
def qnode_edge(v):
    """
    Small circuit used as an "edge-style" descriptor.

    - RY encodes the standardized patch summary.
    - Shallow entangling ring + additional rotations.
    - Output: N_QUBITS expectation values of PauliZ.
    """
    for i in range(N_QUBITS):
        qml.RY(v[i], wires=i)

    for _ in range(QUANTUM_DEPTH):
        # Ring entanglement
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

        # Local re-encoding
        for i in range(N_QUBITS):
            qml.RY(0.5 * v[i], wires=i)
            qml.RZ(0.5 * v[(i + 1) % N_QUBITS], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev_tex, interface="autograd")
def qnode_tex(v):
    """
    "Texture-style" descriptor:

    - Hadamard + RX: more global mixing in amplitudes.
    - Star entanglement around qubit 0.
    - Additional rotations and CZ chain.
    """
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RX(v[i], wires=i)

    for _ in range(QUANTUM_DEPTH):
        # Star topology around qubit 0
        for i in range(1, N_QUBITS):
            qml.CNOT(wires=[0, i])

        for i in range(N_QUBITS):
            qml.RY(0.7 * v[i], wires=i)

        for i in range(N_QUBITS - 1):
            qml.CZ(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# -------------------------------------------------------------------------
# Classical preprocessing for quantum inputs
# -------------------------------------------------------------------------
def extract_qubit_inputs(rgb_np: np.ndarray,
                         q_size: int = QUANTUM_IMAGE_SIZE) -> np.ndarray:
    """
    Compress an RGB image into N_QUBITS scalar values:

    - Convert to grayscale and resize to q_size x q_size.
    - Partition into k x k grid (k ≈ sqrt(N_QUBITS)).
    - Use the mean intensity of each cell.
    - Standardize and scale into a suitable range for rotations.
    """
    pil = Image.fromarray(rgb_np, mode="RGB").convert("L").resize(
        (q_size, q_size), Image.BILINEAR
    )
    arr = np.asarray(pil, dtype=np.float32) / 255.0

    k = int(np.ceil(np.sqrt(N_QUBITS)))
    ph = q_size // k
    pw = q_size // k

    vals = []
    for i in range(k):
        for j in range(k):
            if len(vals) >= N_QUBITS:
                break
            cell = arr[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            vals.append(float(cell.mean()))
        if len(vals) >= N_QUBITS:
            break

    v = np.asarray(vals, dtype=np.float32)
    if len(v) < N_QUBITS:
        v = np.pad(v, (0, N_QUBITS - len(v)), mode="edge")

    # Standardize and limit extreme values before mapping to angles
    v = (v - v.mean()) / (v.std() + 1e-8)
    v = np.clip(v, -3, 3) * (np.pi / 3.0)
    return v.astype(np.float32)


def vec_to_map(v: np.ndarray, target_hw) -> np.ndarray:
    """
    Map a 1D descriptor (length N_QUBITS) to a 2D map,
    then resize to (H, W) to be used as an extra channel.
    """
    n = len(v)
    g = int(np.ceil(np.sqrt(n)))
    pad = g * g - n
    if pad > 0:
        v = np.pad(v, (0, pad), mode="edge")

    grid = np.reshape(v, (g, g))
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    im = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    im = im.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


# -------------------------------------------------------------------------
# Precompute quantum descriptors and cache to disk
# -------------------------------------------------------------------------
def precompute_qvecs(imgs_np,
                     idx_list,
                     out_edge: Path,
                     out_tex: Path,
                     force: bool = False,
                     tag: str = "set") -> None:
    """
    Precompute (edge, texture) quantum descriptors for a subset of images
    and store them as npz files. This is the expensive part, done once.
    """
    if out_edge.exists() and out_tex.exists() and not force:
        print(f"[Precompute] Found {out_edge.name}/{out_tex.name}, skipping.")
        return

    N = len(idx_list)
    qe = np.zeros((N, N_QUBITS), dtype=np.float32)
    qt = np.zeros((N, N_QUBITS), dtype=np.float32)

    print(f"[Precompute] {tag}: {N} images")
    for ii, idx in enumerate(tqdm(idx_list, mininterval=1.0, desc=f"qvec_{tag}")):
        try:
            v = extract_qubit_inputs(imgs_np[idx])
            qe[ii] = np.array(qnode_edge(v), dtype=np.float32)
            qt[ii] = np.array(qnode_tex(v), dtype=np.float32)
        except Exception:
            # In case of numerical / device issues, fall back to zeros
            qe[ii] = 0.0
            qt[ii] = 0.0

    np.savez_compressed(out_edge, qvec=qe)
    np.savez_compressed(out_tex,  qvec=qt)
    print(f"[Precompute] Saved {out_edge.name}, {out_tex.name}")


if PRECOMPUTE_QFEATS:
    precompute_qvecs(train_raw.data, train_idx,
                     TRAIN_Q_EDGE, TRAIN_Q_TEX,
                     FORCE_RECOMPUTE, "train")
    precompute_qvecs(train_raw.data, val_idx,
                     VAL_Q_EDGE, VAL_Q_TEX,
                     FORCE_RECOMPUTE, "val")
    precompute_qvecs(test_raw.data, list(range(len(test_raw))),
                     TEST_Q_EDGE, TEST_Q_TEX,
                     FORCE_RECOMPUTE, "test")

# -------------------------------------------------------------------------
# Dataset wrapper: RGB + 2 quantum maps -> 5-channel input
# -------------------------------------------------------------------------
class QuantumCIFAR100Subset(Dataset):
    """
    CIFAR-100 subset with quantum feature maps.

    Each sample:
      - Regular CIFAR augmentations on RGB.
      - Two additional channels constructed from precomputed quantum descriptors.
      - Total: 5 x H x W input.
    """
    def __init__(self,
                 base_plain,
                 indices,
                 q_edge_npz: Path,
                 q_tex_npz: Path,
                 spatial_tf,
                 is_train: bool):
        self.base = base_plain
        self.indices = list(indices)
        self.spatial_tf = spatial_tf
        self.is_train = is_train

        q_edge = np.load(q_edge_npz)["qvec"]
        q_tex  = np.load(q_tex_npz)["qvec"]

        assert len(q_edge) == len(self.indices)
        assert len(q_tex)  == len(self.indices)

        self.qe = q_edge
        self.qt = q_tex

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img_np, target = self.base.data[idx], int(self.base.targets[idx])

        pil = Image.fromarray(img_np, mode="RGB")
        pil_aug = self.spatial_tf(pil)

        x_rgb = to_tensor(pil_aug)  # 3 x H x W
        H, W = x_rgb.shape[1], x_rgb.shape[2]

        qe_map = vec_to_map(self.qe[i], (H, W))
        qt_map = vec_to_map(self.qt[i], (H, W))

        x_qe = torch.from_numpy(qe_map).unsqueeze(0)  # 1 x H x W
        x_qt = torch.from_numpy(qt_map).unsqueeze(0)  # 1 x H x W

        x5 = torch.cat([x_rgb, x_qe, x_qt], dim=0)    # 5 x H x W
        x5 = norm5(x5)

        if self.is_train:
            x5 = random_erasing(x5)

        return x5, target


train_ds = QuantumCIFAR100Subset(
    train_raw, train_idx, TRAIN_Q_EDGE, TRAIN_Q_TEX, spatial_train, True
)
val_ds = QuantumCIFAR100Subset(
    train_raw, val_idx, VAL_Q_EDGE, VAL_Q_TEX, spatial_val, False
)
test_ds = QuantumCIFAR100Subset(
    test_raw, list(range(len(test_raw))), TEST_Q_EDGE, TEST_Q_TEX, spatial_val, False
)

# Deterministic DataLoaders
g = torch.Generator()
g.manual_seed(SEED)

loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    generator=g,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
)

train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

# -------------------------------------------------------------------------
# Model: ResNet with conv1 expanded to 5 input channels
# -------------------------------------------------------------------------
def expand_resnet_conv1_to_5ch(m: nn.Module) -> nn.Module:
    """
    Replace conv1 to accept 5 channels.
    Initialize the extra channels as the mean over RGB weights.
    """
    old = m.conv1
    new = nn.Conv2d(
        5,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )

    with torch.no_grad():
        new.weight[:, :3, :, :] = old.weight
        mean_rgb = old.weight[:, :3, :, :].mean(dim=1, keepdim=True)
        new.weight[:, 3:4, :, :] = mean_rgb.clone()
        new.weight[:, 4:5, :, :] = mean_rgb.clone()

    m.conv1 = new
    return m


if MODEL_NAME == "resnet18":
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
    model = models.resnet18(weights=weights)
elif MODEL_NAME == "resnet34":
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if PRETRAINED else None
    model = models.resnet34(weights=weights)
elif MODEL_NAME == "resnet50":
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if PRETRAINED else None
    model = models.resnet50(weights=weights)
else:
    raise ValueError("MODEL_NAME must be one of: resnet18 | resnet34 | resnet50")

model = expand_resnet_conv1_to_5ch(model)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# -------------------------------------------------------------------------
# Loss / Optimizer / Scheduler (aligned with baseline)
# -------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

optimizer = optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR_ONECYCLE,
    steps_per_epoch=max(1, len(train_loader)),
    epochs=EPOCHS,
)

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    scheduler_step="batch",
    num_epochs=EPOCHS,
    device=device,
    is_binary=False,
    save_metric="f1",
    output_dir=str(RUN_DIR),
    f1_average="macro",
    problem_type="multiclass",
    num_classes=NUM_CLASSES,
    class_names=CLASS_NAMES,
)

# -------------------------------------------------------------------------
# Evaluation on test set
# -------------------------------------------------------------------------
@torch.no_grad()
def eval_top1(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


acc = eval_top1(trained_model, test_loader, device=device)
print(f"[Hybrid 5-ch] Final CIFAR-100 Test Top-1: {acc:.4f}")

save_path = RUN_DIR / "final_model_cifar100_quantum_hybrid.pth"
torch.save(trained_model.state_dict(), save_path)
print("Saved:", save_path)
