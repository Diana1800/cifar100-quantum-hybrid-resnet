import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

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
from pennylane import qnn as qnn_pl

from DLfunctions import train_model

# -------------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # deterministic cuBLAS

SEED          = 42
SPLIT_SEED    = 42
DATA_DIR      = "./data"
RUN_DIR       = Path("./runs/cifar100_hybrid_lastQ_fair_alpha_det")
SPLIT_PATH    = Path("./runs/cifar100_baseline_det/split_idx_cifar100.npz")

MODEL_NAME    = "resnet34"
PRETRAINED    = True
IMG_SIZE      = 224
EPOCHS        = 40
EPOCHS_WARMUP = 5  # Phase A: quantum effectively off, Phase B: quantum on

BATCH_SIZE    = 128
NUM_WORKERS   = 0   # strict reproducibility
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
MAX_LR_ONECYCLE = 7e-4
LABEL_SMOOTH  = 0.05

# Quantum config
N_QUBITS        = 4
N_LAYERS        = 2
DIFF_METHOD     = "adjoint"
Q_TORCH_DEVICE  = torch.device("cpu")  # TorchLayer currently on CPU

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
# Transforms (aligned with baseline)
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
norm3 = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
random_erasing = transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3))

# -------------------------------------------------------------------------
# CIFAR-100 & split (reuse baseline split)
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
# Dataset wrappers (RGB only; identical to baseline)
# -------------------------------------------------------------------------
class CIFAR100SubsetRGB(Dataset):
    """
    Simple CIFAR-100 subset wrapper with spatial augmentations and RGB normalization.
    """
    def __init__(self, base_plain, indices, spatial_tf, is_train: bool):
        self.base = base_plain
        self.indices = list(indices)
        self.spatial_tf = spatial_tf
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img_np, target = self.base.data[idx], int(self.base.targets[idx])

        pil = Image.fromarray(img_np, mode="RGB")
        pil_aug = self.spatial_tf(pil)

        x = to_tensor(pil_aug)
        x = norm3(x)

        if self.is_train:
            x = random_erasing(x)

        return x, target


train_ds = CIFAR100SubsetRGB(train_raw, train_idx, spatial_train, True)
val_ds   = CIFAR100SubsetRGB(train_raw, val_idx,   spatial_val,   False)
test_ds  = CIFAR100SubsetRGB(test_raw, list(range(len(test_raw))), spatial_val, False)

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
# Base ResNet backbone (identical to baseline)
# -------------------------------------------------------------------------
if MODEL_NAME == "resnet18":
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
    base_model = models.resnet18(weights=weights)
elif MODEL_NAME == "resnet34":
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if PRETRAINED else None
    base_model = models.resnet34(weights=weights)
elif MODEL_NAME == "resnet50":
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if PRETRAINED else None
    base_model = models.resnet50(weights=weights)
else:
    raise ValueError("MODEL_NAME must be resnet18 | resnet34 | resnet50")

# -------------------------------------------------------------------------
# Quantum head (PennyLane TorchLayer)
# -------------------------------------------------------------------------
dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def quantum_circuit(inputs, weights):
    """
    Variational circuit:

    - AngleEmbedding encodes the projected backbone features.
    - StronglyEntanglingLayers provide a standard expressive ansatz.
    - Output: N_QUBITS expectation values of PauliZ.
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class HybridResNetLastQ(nn.Module):
    """
    ResNet backbone + classical head + quantum head with learnable fusion:

        logits = logits_cls + alpha * logits_q

    where alpha = sigmoid(alpha_raw) is a scalar in (0, 1).
    """
    def __init__(self,
                 base_model: nn.Module,
                 num_classes: int,
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 alpha_init_raw: float = -20.0):
        super().__init__()

        # Backbone: all layers except the final fc
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.flatten = nn.Flatten()
        in_feat = base_model.fc.in_features

        # Classical head: same capacity as the baseline
        self.classical_head = nn.Linear(in_feat, num_classes)

        # Quantum branch:
        #   in_feat -> n_qubits (linear projection) -> circuit -> n_qubits -> num_classes
        self.pre_q = nn.Linear(in_feat, n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_head = qnn_pl.TorchLayer(quantum_circuit, weight_shapes)
        self.post_q = nn.Linear(n_qubits, num_classes)

        # Fusion parameter: alpha = sigmoid(alpha_raw)
        # alpha_init_raw = -20.0 => alpha ~ 2e-9 (nearly zero at warm-up)
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init_raw))

        # Explicit placement for the torch quantum head
        self.q_torch_device = Q_TORCH_DEVICE

    # --- helpers for warm-up / phase scheduling ---
    def set_alpha_raw(self, v: float, trainable: bool) -> None:
        with torch.no_grad():
            self.alpha_raw.copy_(torch.tensor(float(v), device=self.alpha_raw.device))
        self.alpha_raw.requires_grad_(trainable)

    def freeze_quantum(self, freeze: bool) -> None:
        for p in self.pre_q.parameters():
            p.requires_grad = not freeze
        for p in self.q_head.parameters():
            p.requires_grad = not freeze
        for p in self.post_q.parameters():
            p.requires_grad = not freeze

    def freeze_backbone(self, freeze: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        for p in self.classical_head.parameters():
            p.requires_grad = not freeze

    # --- forward ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        feats = self.flatten(self.backbone(x))           # (B, C)
        logits_cls = self.classical_head(feats)          # (B, num_classes)

        # Quantum branch
        q_in = self.pre_q(feats)                         # (B, n_qubits) on GPU
        q_in_cpu = q_in.to(self.q_torch_device)          # move to CPU for TorchLayer
        q_feat_cpu = self.q_head(q_in_cpu)               # (B, n_qubits) on CPU
        q_feat = q_feat_cpu.to(feats.device)             # back to GPU
        logits_q = self.post_q(q_feat)                   # (B, num_classes)

        alpha = torch.sigmoid(self.alpha_raw)            # scalar in (0, 1)
        return logits_cls + alpha * logits_q


print(f"Creating Hybrid model with last-layer quantum: {N_QUBITS} qubits, {N_LAYERS} layers.")
model = HybridResNetLastQ(
    base_model,
    num_classes=NUM_CLASSES,
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    alpha_init_raw=-20.0,
).to(device)

# -------------------------------------------------------------------------
# Loss / Optimizer / Scheduler
# -------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)


def make_optim_sched(model: nn.Module, epochs: int):
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR_ONECYCLE,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs,
    )
    return optimizer, scheduler


# -------------------------------------------------------------------------
# Utility: print alpha once per epoch in Phase B
# -------------------------------------------------------------------------
_epoch_batch_counter = {"count": 0}


def make_alpha_epoch_printer(model: HybridResNetLastQ, steps_per_epoch: int):
    def pre_hook(_module, _inputs):
        if not model.training:
            return
        _epoch_batch_counter["count"] += 1
        b = _epoch_batch_counter["count"] - 1  # zero-based
        if b % steps_per_epoch == 0:
            current_epoch = b // steps_per_epoch + 1
            with torch.no_grad():
                a = torch.sigmoid(model.alpha_raw).item()
            print(f"[Phase B] Epoch {current_epoch}: alpha(sigmoid) = {a:.6f}")
    return pre_hook


# -------------------------------------------------------------------------
# Training: Phase A (warm-up) + Phase B (quantum enabled)
# -------------------------------------------------------------------------
if EPOCHS_WARMUP > 0:
    # ---- Phase A: warm-up (quantum effectively off, backbone fully trainable) ----
    print(
        f"\n=== Phase A: Warm-up for {EPOCHS_WARMUP} epochs "
        f"(alpha≈0, quantum frozen, backbone trainable) ==="
    )
    model.set_alpha_raw(-20.0, trainable=False)  # alpha ~ 0
    model.freeze_quantum(True)
    model.freeze_backbone(False)

    optA, schA = make_optim_sched(model, epochs=EPOCHS_WARMUP)
    trained_model, histA = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optA,
        scheduler=schA,
        scheduler_step="batch",
        num_epochs=EPOCHS_WARMUP,
        device=device,
        is_binary=False,
        save_metric="f1",
        output_dir=str(RUN_DIR),
        f1_average="macro",
        problem_type="multiclass",
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
    )

    # ---- Phase B: quantum fusion (alpha learnable, quantum unfrozen) ----
    print(
        f"\n=== Phase B: Quantum fusion for {EPOCHS - EPOCHS_WARMUP} epochs "
        f"(alpha learnable, quantum unfrozen, backbone trainable) ==="
    )
    model.set_alpha_raw(0.0, trainable=True)  # alpha ≈ 0.5 at start (can be changed)
    model.freeze_quantum(False)
    model.freeze_backbone(False)

    steps_per_epoch = max(1, len(train_loader))
    hook_handle = model.register_forward_pre_hook(
        make_alpha_epoch_printer(model, steps_per_epoch)
    )

    optB, schB = make_optim_sched(model, epochs=EPOCHS - EPOCHS_WARMUP)
    trained_model, histB = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optB,
        scheduler=schB,
        scheduler_step="batch",
        num_epochs=EPOCHS - EPOCHS_WARMUP,
        device=device,
        is_binary=False,
        save_metric="f1",
        output_dir=str(RUN_DIR),
        f1_average="macro",
        problem_type="multiclass",
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
    )

    hook_handle.remove()

else:
    # Single-phase training with quantum from the start
    print(
        f"\n=== Single-phase: Quantum fused from start for {EPOCHS} epochs "
        f"(alpha learnable) ==="
    )
    model.set_alpha_raw(0.0, trainable=True)
    model.freeze_quantum(False)
    model.freeze_backbone(False)

    steps_per_epoch = max(1, len(train_loader))
    hook_handle = model.register_forward_pre_hook(
        make_alpha_epoch_printer(model, steps_per_epoch)
    )

    opt, sch = make_optim_sched(model, epochs=EPOCHS)
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=opt,
        scheduler=sch,
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

    hook_handle.remove()

# -------------------------------------------------------------------------
# Final test (same metric as baseline)
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
print(f"[Hybrid ResNet34 + Quantum Head] Final CIFAR-100 Test Top-1: {acc:.4f}")

save_path = RUN_DIR / "final_model_cifar100_hybrid_lastQ_fair_alpha.pth"
torch.save(model.state_dict(), save_path)

with torch.no_grad():
    final_alpha = torch.sigmoid(model.alpha_raw).item()

print(f"Saved: {save_path} | Final alpha(sigmoid)={final_alpha:.6f}")
