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

from DLfunctions import train_model  # your existing helper

# ----------------------------------------------------------------------
# Configuration (aligned with hybrid runs)
# ----------------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # deterministic cuBLAS

SEED        = 42
SPLIT_SEED  = 42
DATA_DIR    = "./data"
RUN_DIR     = Path("./runs/cifar100_baseline_det")
RUN_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_PATH  = RUN_DIR / "split_idx_cifar100.npz"

MODEL_NAME  = "resnet34"
PRETRAINED  = True
IMG_SIZE    = 224
EPOCHS      = 40
BATCH_SIZE  = 128
NUM_WORKERS = 0

LR              = 3e-4
WEIGHT_DECAY    = 1e-4
MAX_LR_ONECYCLE = 7e-4
LABEL_SMOOTH    = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_all_seeds(SEED)

# ----------------------------------------------------------------------
# Data / Transforms
# ----------------------------------------------------------------------
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
random_erasing = transforms.RandomErasing(
    p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)
)

train_raw = datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
test_raw  = datasets.CIFAR100(root=DATA_DIR, train=False, download=True)

CLASS_NAMES = train_raw.classes
NUM_CLASSES = 100

# ----------------------------------------------------------------------
# Split (shared with hybrid experiments)
# ----------------------------------------------------------------------
if SPLIT_PATH.exists():
    d = np.load(SPLIT_PATH)
    train_idx = d["train_idx"].tolist()
    val_idx   = d["val_idx"].tolist()
    print(f"Loaded split from {SPLIT_PATH}")
else:
    print(f"Creating deterministic split at {SPLIT_PATH}")
    rng = random.Random(SPLIT_SEED)
    N = len(train_raw)
    idx = list(range(N))
    rng.shuffle(idx)
    n_val = int(0.2 * N)
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])
    np.savez_compressed(
        SPLIT_PATH,
        train_idx=np.asarray(train_idx),
        val_idx=np.asarray(val_idx),
    )

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_raw)}")

# ----------------------------------------------------------------------
# Dataset wrappers
# ----------------------------------------------------------------------
class CIFAR100SubsetRGB(Dataset):
    def __init__(self, base_plain, indices, spatial_tf, is_train: bool):
        self.base = base_plain
        self.indices = list(indices)
        self.spatial_tf = spatial_tf
        self.is_train = is_train

    def __len__(self):
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

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    generator=g,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    generator=g,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
)

# ----------------------------------------------------------------------
# Model: standard ResNet-34 baseline
# ----------------------------------------------------------------------
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
    raise ValueError("MODEL_NAME must be resnet18|resnet34|resnet50")

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# ----------------------------------------------------------------------
# Loss / Optimizer / Scheduler
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Final evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_top1(model, loader, device="cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


acc = eval_top1(trained_model, test_loader, device=device)
print(f"[Baseline ResNet34] Final CIFAR-100 Test Top-1: {acc:.4f}")

save_path = RUN_DIR / "final_model_resnet34_baseline.pth"
torch.save(trained_model.state_dict(), save_path)
print("Saved:", save_path)
