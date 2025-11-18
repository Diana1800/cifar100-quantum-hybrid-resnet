from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from sklearn.metrics import roc_auc_score, roc_curve
import time
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.amp import autocast, GradScaler
from collections import OrderedDict
import torch.nn.functional as F


# Auto-detect problem type based on outputs/labels shape & dtype
def infer_problem_type(outputs, labels):
    out_ndim = outputs.ndim
    out_last = outputs.shape[-1] if out_ndim > 1 else 1
    lab_ndim = labels.ndim

    # multilabel: labels [N,C] float/bool
    if lab_ndim == 2 and out_ndim == 2 and out_last > 1 and labels.dtype.is_floating_point:
        return "multilabel"

    # multiclass: outputs [N,C>=3], labels [N] int
    if out_ndim == 2 and out_last >= 3 and lab_ndim == 1 and labels.dtype in (torch.long, torch.int64):
        return "multiclass"

    # binary_ce_2logit: outputs [N,2], labels [N] int
    if out_ndim == 2 and out_last == 2 and lab_ndim == 1 and labels.dtype in (torch.long, torch.int64):
        return "binary_ce_2logit"

    # binary_bce_1logit: outputs [N,1] or [N], labels [N] float
    if (out_ndim in (1,2) and out_last == 1) and lab_ndim == 1 and labels.dtype.is_floating_point:
        return "binary_bce_1logit"

    # fallback
    return "multiclass"

# Convert raw outputs -> (probs, preds, labels) on CPU
@torch.no_grad()
def postprocess_outputs(outputs, labels, problem_type: str, decision_threshold: float = 0.5):

    if problem_type == "multiclass":
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        return probs.cpu(), preds.cpu(), labels.cpu()

    if problem_type == "binary_ce_2logit":
        probs_pos = F.softmax(outputs, dim=1)[:, 1]
        preds = (probs_pos >= decision_threshold).long()
        return probs_pos.cpu(), preds.cpu(), labels.cpu()

    if problem_type == "binary_bce_1logit":
        logits = outputs.flatten()
        probs_pos = torch.sigmoid(logits)
        preds = (probs_pos >= decision_threshold).long()
        return probs_pos.cpu(), preds.cpu(), labels.cpu()

    if problem_type == "multilabel":
        probs = torch.sigmoid(outputs)
        preds = (probs >= decision_threshold).to(torch.uint8)
        return probs.cpu(), preds.cpu(), labels.cpu()

    raise ValueError(f"Unknown problem_type: {problem_type}")



# robust scalar converter so we can safely format/record metrics
def _to_scalar(x):
    # TorchMetrics Metric -> compute()
    if hasattr(x, "compute") and callable(x.compute):
        x = x.compute()
    # torch.Tensor -> item()
    if isinstance(x, torch.Tensor):
        try:
            return float(x.detach().cpu().item())
        except Exception:
            x = x.detach().cpu()
            try:
                return float(x)
            except Exception:
                return x
    try:
        return float(x)
    except Exception:
        return x  # last resort (don't printf with :.4f)
    
# ======================= Freezing functions ===========================   

def freeze_model_layers(model, layer_patterns):
    """
    FREEZE specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_patterns: List of strings/patterns to match layer names
                       e.g., ['backbone', 'features.0', 'layer1', 'layer2']
    
    Example usage:
        freeze_model_layers(model, ['layer1', 'layer2', 'layer3'])  # Freeze early ResNet layers
    """
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if any(pattern in name for pattern in layer_patterns):
            param.requires_grad = False
            frozen_params += param.numel()
            print(f"FROZEN: {name} ({param.numel():,} params)")
    
    print(f"FROZEN {frozen_params:,} / {total_params:,} parameters ({100 * frozen_params / total_params:.1f}%)")


def unfreeze_model_layers(model, layer_patterns):
    """
    UNFREEZE specific layers in a model (for progressive unfreezing).
    
    Args:
        model: PyTorch model
        layer_patterns: List of strings/patterns to match layer names
    """
    unfrozen_params = 0
    
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in layer_patterns):
            param.requires_grad = True
            unfrozen_params += param.numel()
            print(f"UNFROZEN: {name} ({param.numel():,} params)")
    
    print(f"UNFROZEN {unfrozen_params:,} parameters")


def set_requires_grad_by_prefix(model, prefixes, requires_grad=True):                                       # include all layers whose names start with any of the given prefixes
    tup = tuple(prefixes)
    for n, p in model.named_parameters():
        if n.startswith(tup):
            p.requires_grad = requires_grad

def build_optimizer(params, optimizer_name="adamw", lr=1e-3, weight_decay=1e-4, **kw):                      # Create optimizer by name if needed
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kw)
    if optimizer_name == "sgd":
        momentum = kw.pop("momentum", 0.9)
        nesterov = kw.pop("nesterov", True)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, **kw)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kw)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def auto_discover_layer_groups(model):                     
    """
    Return an ordered list of parameter name groups for progressive unfreezing.
    For torchvision ResNets: [conv1+bn1, layer1, layer2, layer3, layer4, fc]
    Falls back to a single group (all non-FC) + FC if structure is unknown.
    """
    groups = []
    # Try ResNet-style blocks
    names = [n for n, _ in model.named_parameters()]
    has = lambda key: any(key in n for n in names)

    if has("layer1") and has("layer2") and has("layer3") and has("layer4"):
        g0 = [n for n in names if n.startswith(("conv1", "bn1"))]
        g1 = [n for n in names if n.startswith("layer1")]
        g2 = [n for n in names if n.startswith("layer2")]
        g3 = [n for n in names if n.startswith("layer3")]
        g4 = [n for n in names if n.startswith("layer4")]
        g5 = [n for n in names if n.startswith(("fc.", "fc_weight", "fc_bias")) or n == "fc.weight" or n == "fc.bias"]
        # Only add non-empty groups
        groups = [g for g in [g0, g1, g2, g3, g4, g5] if g]
    else:
        # Generic fallback: everything-not-fc, then fc
        non_fc = [n for n in names if not (n.startswith("fc.") or n in ("fc.weight","fc.bias"))]
        fc     = [n for n in names if     (n.startswith("fc.") or n in ("fc.weight","fc.bias"))]
        groups = [non_fc, fc] if fc else [non_fc]

    return groups


def auto_discover_layer_groups_from_names(param_names):
    """
    Generic fallback: cluster parameter names by their top-level prefix before the first dot.
    E.g. 'layer3.0.conv1.weight' -> 'layer3'
    Returns a coarse early->late ordering.
    """
    buckets = OrderedDict()
    for n in param_names:
        top = n.split('.', 1)[0]
        buckets.setdefault(top, []).append(n)
    # heuristic: put classifier heads last if present
    ordered = []
    tail = []
    for k, v in buckets.items():
        if k in ("fc", "classifier", "head"):
            tail.append(v)
        else:
            ordered.append(v)
    return ordered + tail

def get_model_layer_groups(model, model_type='auto'):
    """
    Return ordered name-prefix groups for progressive unfreezing.
    """
    model_name = model.__class__.__name__.lower() if model_type == 'auto' else model_type.lower()
    if model_type == 'auto':
        if 'resnet' in model_name:        model_type = 'resnet'
        elif 'vgg' in model_name:         model_type = 'vgg'
        elif 'densenet' in model_name:    model_type = 'densenet'
        elif 'efficientnet' in model_name:model_type = 'efficientnet'
        elif 'vit' in model_name or 'transformer' in model_name: model_type = 'vit'
        elif 'bert' in model_name:        model_type = 'bert'
        else:                              model_type = 'generic'

    # explicit patterns for common families
    if model_type == 'resnet':
        groups = [
            ['conv1', 'bn1'],
            ['layer1'],
            ['layer2'],
            ['layer3'],
            ['layer4'],
            ['fc', 'classifier'],
        ]
    elif model_type == 'vgg':
        groups = [
            ['features.0','features.1','features.2'],
            ['features.3','features.4','features.5'],
            ['features.6','features.7','features.8'],
            ['features.9','features.10'],
            ['classifier'],
        ]
    elif model_type == 'densenet':
        groups = [
            ['features.conv0','features.norm0'],
            ['features.denseblock1'],
            ['features.denseblock2'],
            ['features.denseblock3'],
            ['features.denseblock4'],
            ['classifier'],
        ]
    elif model_type == 'efficientnet':
        groups = [
            ['features.0'],
            ['features.1','features.2'],
            ['features.3','features.4'],
            ['features.5','features.6'],
            ['features.7','features.8'],
            ['classifier'],
        ]
    elif model_type == 'vit':
        groups = [
            ['patch_embed'],
            ['blocks.0','blocks.1','blocks.2'],
            ['blocks.3','blocks.4','blocks.5'],
            ['blocks.6','blocks.7','blocks.8'],
            ['blocks.9','blocks.10','blocks.11'],
            ['head','classifier'],
        ]
    elif model_type == 'bert':
        groups = [
            ['embeddings'],
            ['encoder.layer.0','encoder.layer.1','encoder.layer.2'],
            ['encoder.layer.3','encoder.layer.4','encoder.layer.5'],
            ['encoder.layer.6','encoder.layer.7','encoder.layer.8'],
            ['encoder.layer.9','encoder.layer.10','encoder.layer.11'],
            ['classifier','pooler'],
        ]
    else:
        # generic fallback from actual names
        names = [n for n, _ in model.named_parameters()]
        grouped = auto_discover_layer_groups_from_names(names)
        return grouped

    # filter out empty groups (depending on the exact arch)
    names = [n for n, _ in model.named_parameters()]
    filtered = []
    for prefixes in groups:
        bucket = [n for n in names if any(n.startswith(pfx) for pfx in prefixes)]
        if bucket:
            filtered.append(bucket)
    return filtered



# ======================= Plotting functions ===========================

def plot_gradient_and_lr(gradient_norms, learning_rates, epoch, output_dir):
    if not gradient_norms:  # Skip if no data
        return
    plt.figure(figsize=(14, 5))

    # Plot the Gradient Norms
    plt.subplot(1, 2, 1)
    plt.plot(gradient_norms, label='Gradient Norms')
    plt.title(f'Gradient Norms for Epoch {epoch + 1}')
    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.grid(True)

    # Plot the Learning Rates
    plt.subplot(1, 2, 2)
    if learning_rates:
        plt.plot(learning_rates, label='Learning Rate', color='orange')
    plt.title(f'Learning Rate for Epoch {epoch + 1}')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir /f'gradient_lr_epoch_{epoch + 1}.png')  
    plt.close() 


def plot_loss(history, epoch, output_dir):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    epochs_so_far = list(range(1, len(history['train_loss']) + 1))
    plt.plot(epochs_so_far, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs_so_far, history['val_loss'], label='Validation Loss', color='red')
    plt.title(f'Loss Progress through Epoch {epoch + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_dir / f'loss_epoch_{epoch + 1}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, epoch, output_dir, is_binary=True, class_names=None):
    if is_binary:
        cm = confusion_matrix(y_true, y_pred)
        fmt = "d"; annot = True; figsize = (8, 6)
        title = f'Confusion Matrix (Binary) - Epoch {epoch+1}'
    else:
        cm = confusion_matrix(y_true, y_pred, normalize='true')  # rows sum to 1
        fmt = ".2f"; annot = False; figsize = (20, 20)
        title = f'Confusion Matrix (Multiclass, row-normalized) - Epoch {epoch+1}'

    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=annot, fmt=fmt, cmap="Blues", cbar=True)
    if not is_binary and class_names:
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, rotation=0,  fontsize=6)
    plt.title(title)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_epoch_{epoch+1}.png')
    plt.close()



# for multiclass we plot max confidence (not class-1 prob)
def plot_prob_distribution(probs, epoch, output_dir, is_binary):
    if not probs:  # Skip if no data
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(probs, bins=30, kde=True, color='purple')
    title = 'Predicted Probability' if is_binary else 'Max Softmax Confidence'
    plt.title(f'Distribution of {title} for Epoch {epoch + 1}')
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_dir /f'prob_distribution_epoch_{epoch + 1}.png')
    plt.close()



# ======================= Training function ===========================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, scheduler_step="batch", num_epochs=10, device='cuda', is_binary=True, save_metric='f1', l1_lambda=None, output_dir="outputs", oTBLogger=None, f1_average=None, problem_type: str = "multiclass", num_classes: int = None, decision_threshold: float = 0.5, use_amp: bool = False, max_grad_norm=None, 
    resume_from_checkpoint=None, freeze_layers=None, class_names=None, log_every_n_batches=50, unfreeze_schedule=None, progressive_unfreeze=False, freeze_layer_groups=None, freeze_optimizer_name="adamw", freeze_base_lr=None, freeze_base_weight_decay=None):

    """
    Universal training loop supporting binary, multiclass, and multilabel classification.
    
    Args:
        model: PyTorch model
        train_loader, val_loader: DataLoaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        scheduler_step: "batch", "epoch", or "plateau"
        num_epochs: Number of training epochs
        device: Device to train on
        save_metric: "loss" or "f1" - metric to use for saving best model
        l1_lambda: L1 regularization strength (optional)
        output_dir: Directory to save outputs
        f1_average: "binary", "macro", "micro", "weighted" for F1 computation
        problem_type: Force specific problem type (optional, auto-detected otherwise)
        decision_threshold: Threshold for binary predictions
        use_amp: Whether to use mixed precision training
        class_names: List of class names for plotting (optional)
        log_every_n_batches: Print progress every N batches
        unfreeze_schedule: Dict of {epoch: [layer_prefixes]} for progressive unfreezing
        progressive_unfreeze: Enable progressive unfreezing
        freeze_optimizer_name: Optimizer type for unfrozen layers ("adamw", "sgd", "adam")
        freeze_base_lr: Learning rate when recreating optimizer for unfreezing
        freeze_base_weight_decay: Weight decay for unfreezing optimizer
    
    Returns:
        model: Trained model (loaded with best weights)
        history: Dictionary containing training metrics
    """

    history = {'train_loss': [], 'val_loss': [], 'train_score': [], 'val_score': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'roc_auc': [], 'epoch_time': [], 'learning_rate': [], 'batch_learning_rate': [], 'gradient_norms': [],
        'last_saved_epoch_loss': None, 'last_saved_epoch_f1': None}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_val_f1 = float('-inf')  # Assuming F1 score ranges between 0 and 1
    best_model_wts_loss = None
    best_model_wts_f1   = None
    last_saved_epoch_loss = 0
    last_saved_epoch_f1 = 0
    start_epoch = 0
    acc_binary = None  
    acc_multi  = None
    
    model = model.to(device)



    # Initialize Layer Freezing if specified
    if freeze_layers:
        print(f"Initialize Layer Freezing: {freeze_layers}")
        freeze_model_layers(model, freeze_layers)                                    # Freeze specified layers at start of training

    if progressive_unfreeze and freeze_layer_groups is None:
        freeze_layer_groups = get_model_layer_groups(model, model_type='auto')       # Auto-detect layer groups if not provided

    # Mixed precision scaler - AMP - mix float16 and float32 ops to save memory and speed up training on gpu
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)      # GradScaler to manage scaling of gradients for AMP - strech dynamic range of float16 (to avoid underflow to zero)

    # Resume from checkpoint if provided
    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_f1 = checkpoint.get('best_val_f1', float('-inf'))
        problem_type = checkpoint.get('problem_type', problem_type)
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"RESUMED from epoch {start_epoch}, problem_type: {problem_type}")




    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        if progressive_unfreeze and isinstance(unfreeze_schedule, dict):
            if epoch in unfreeze_schedule:
                prefixes = unfreeze_schedule[epoch]  # list of name prefixes
                set_requires_grad_by_prefix(model, prefixes, True)
                if freeze_optimizer_name:
                    lr = freeze_base_lr if freeze_base_lr is not None else 1e-3
                    wd = freeze_base_weight_decay if freeze_base_weight_decay is not None else 1e-4
                    trainable_params = (p for p in model.parameters() if p.requires_grad)
                    optimizer = build_optimizer(trainable_params, optimizer_name=freeze_optimizer_name, lr=lr, weight_decay=wd)
                 
                

        # =========================================================== Training Phase ===========================================================

        model.train()  # training mode
        train_losses = []
        all_train_labels = []
        all_train_preds = []
        current_epoch_gradient_norms = []  # Track gradient norms for the current epoch
        current_epoch_lrs = []  # Track learning rates for the current epoch
        all_train_probs = [] #To Plot the distribution of predicted probabilities
        
        
        total_train_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)                 # Move data to device - non_blocking=True for async transfer if pinned memory (ignored if not pinned)- (cpu pin memory when dataloader to speeds up transfer to gpu as no staging copy needed)
            optimizer.zero_grad(set_to_none=True)                         # Clear gradients - set_to_none=True is more efficient as it avoids unnecessary memory ops, we zero only the gradients that will be updated

            # Forward 
            with torch.amp.autocast("cuda", enabled=use_amp):      # Autocast context for AMP (if enabled) - in autocast gradients from FP16 can underflow(become zero) - so we use scaler(GradScaler) to bring to normal range, FP32
                outputs = model(images)
                loss = criterion(outputs, labels)                      # Compute loss - by CrossEntropy(multi-class)/ BCE(binary)/MSE(regrssion) etc if in optimizer we use weight decay (L2 regularization)
            if l1_lambda:                                              # L1 regularization - if added later with optimizer to L2 loss (if in optimizer we use weight decay (L2 regularization)) - we get Elastic net style
                loss = loss + l1_lambda * sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

            # Backward
            if use_amp:                                  # Automatic Mixed Precision - training in float16 where safe and float32 where needed - Faster training and less memory
                scaler.scale(loss).backward()            
                scaler.unscale_(optimizer)                
                if max_grad_norm is not None:            # Gradient clipping - cap the max norm  - to avoid gradient explosion 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)      
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Auto-detect problem type on first batch
            if problem_type is None:
                problem_type = infer_problem_type(outputs, labels)
                is_binary = problem_type in ["binary_ce_2logit", "binary_bce_1logit"]
                print(f"Auto-detected problem type: {problem_type}")

            if acc_binary is None and (problem_type in ["binary_ce_2logit", "binary_bce_1logit"]):
                acc_binary = BinaryAccuracy(threshold=decision_threshold).to(device)
            if acc_multi is None and (problem_type == "multiclass"):
                pass  

            # Post-process outputs to get probs, preds, labels on CPU
            probs_cpu, preds_cpu, labels_cpu = postprocess_outputs(
                outputs, labels, problem_type=problem_type, decision_threshold=decision_threshold)

            # Store lists (1D vs 2D safe)
            all_train_labels.extend(labels_cpu.view(-1).numpy().tolist())
            if probs_cpu.ndim == 1:
                all_train_probs.extend(probs_cpu.numpy().tolist())
            else:
                # multiclass/multilabel: store max-confidence only for the prob-distribution plot
                max_conf = probs_cpu.max(dim=1).values
                all_train_probs.extend(max_conf.numpy().tolist())
            if preds_cpu.ndim == 1:
                all_train_preds.extend(preds_cpu.numpy().tolist())
            else:
                all_train_preds.extend((preds_cpu.sum(dim=1) > 0).numpy().astype(int).tolist())


            # per-batch scheduler
            if scheduler is not None and scheduler_step == "batch":
                scheduler.step()
                current_lr = float(optimizer.param_groups[0]['lr'])
                current_epoch_lrs.append(current_lr)
                history['batch_learning_rate'].append(current_lr)
            
            # ====== Monitoring Prints ===========   
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += (p.grad.data.norm(2).item())**2            # L2 norm - sum of squares of all gradients
            total_norm = total_norm ** 0.5                                   # Global norm - squer root of sum of squares
            current_epoch_gradient_norms.append(total_norm)
            history['gradient_norms'].append(total_norm)

            train_losses.append(float(loss.item()))

            # Prints - Batch progress (guard cuda prints)
            if (batch_idx + 1) % log_every_n_batches == 0 or batch_idx == 0:
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                    reserv = torch.cuda.memory_reserved() / (1024 ** 2)
                    print(f'\rTrain Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{total_train_batches} - '
                        f'Loss: {loss.item():.4f} | GradNorm: {total_norm:.4f} | '
                        f'Allocated: {alloc:.2f} MB | Reserved: {reserv:.2f} MB', end='')
                else:
                    print(f'\rTrain Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{total_train_batches} - '
                        f'Loss: {loss.item():.4f} | GradNorm: {total_norm:.4f}', end='')
                print('', end='\r')


        # ================ Train Epoch Metrics Calculation =================
        
        train_loss = float(np.mean(train_losses))
           
        # Train Score, Precision, Recall, F1
        train_score = accuracy_score(all_train_labels, all_train_preds)
        average_method = 'binary' if is_binary else ('macro' if f1_average is None else f1_average)
        train_precision = precision_score(all_train_labels, all_train_preds, average=average_method, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average=average_method, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average=average_method, zero_division=0)








        # =========================================================== Validation Phase ===========================================================

        model.eval()
        val_losses = []
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []
        all_val_probs_full = []  # for multiclass AUC

        
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_losses.append(float(loss.item()))

                probs_cpu, preds_cpu, labels_cpu = postprocess_outputs(
                    outputs, labels, problem_type=problem_type, decision_threshold=decision_threshold)

                # all_val_labels.extend(labels_cpu.view(-1).numpy().tolist())
                # if probs_cpu.ndim == 1:
                #     all_val_probs.extend(probs_cpu.numpy().tolist())
                # else:
                #     max_conf = probs_cpu.max(dim=1).values
                #     all_val_probs.extend(max_conf.numpy().tolist())
                all_val_labels.extend(labels_cpu.view(-1).numpy().tolist())

                if probs_cpu.ndim == 1:
                    all_val_probs.extend(probs_cpu.numpy().tolist())         # binary
                else:
                    max_conf = probs_cpu.max(dim=1).values
                    all_val_probs.extend(max_conf.numpy().tolist())         # for prob-distribution plot
                    all_val_probs_full.append(probs_cpu.numpy())           

                if preds_cpu.ndim == 1:
                    all_val_preds.extend(preds_cpu.numpy().tolist())
                else:
                    all_val_preds.extend((preds_cpu.sum(dim=1) > 0).numpy().astype(int).tolist())

                # Progress logging for validation
                if (batch_idx + 1) % log_every_n_batches == 0 or batch_idx == 0:
                    print(f'\rValidation Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{total_val_batches} - '
                          f'Loss: {loss.item():.4f}', end='')
        print()    #('', end='\r') 

        val_loss = float(np.mean(val_losses))

        # Once per epoch
        if scheduler is not None:
            if scheduler_step == "epoch":
                scheduler.step()               # e.g., CosineAnnealingLR, StepLR, MultiStepLR, ExponentialLR
            elif scheduler_step == "plateau":
                scheduler.step(val_loss)       # ReduceLROnPlateau needs val metric
       
       
        # ================= Val Epoch Metrics Calculation =================

        # Val Score, Precision, Recall, F1 
        val_score = accuracy_score(all_val_labels, all_val_preds)  
        val_precision = precision_score(all_val_labels, all_val_preds, average=average_method, zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average=average_method, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average=average_method, zero_division=0)

        # ROC-AUC (binary only)
        roc_auc = None          # Accept both shapes: (N,) or (N,2) and extract p(positive)
        if is_binary:
            val_probs = np.asarray(all_val_probs)
            if val_probs.ndim == 2 and val_probs.shape[1] == 2:                 # (N,2) shape - probs for both classes
                y_scores = val_probs[:, 1]                                      # prob of class 1
            else:
                y_scores = val_probs.reshape(-1)                                # already positive-class probs

            y_true = np.asarray(all_val_labels).astype(int)

            # Guard against degenerate epoch (only one class present in y_true)
            roc_auc = None
            try:
                    # require both classes to exist to avoid exceptions
                if np.unique(y_true).size == 2:
                    roc_auc = roc_auc_score(y_true, y_scores)
                    fpr, tpr, _ = roc_curve(y_true, y_scores)

                    plt.figure(figsize=(6, 6))
                    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
                    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
                    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve - Epoch {epoch + 1}")
                    plt.legend(loc="lower right")
                    plt.grid(True)
                    plt.savefig(output_dir / f"roc_curve_epoch_{epoch + 1}.png")
                    plt.close()
            except Exception as e:
                roc_auc = None  # keep training; some epochs may be degenerate
        # ROC-AUC (multiclass OvR, macro)
        if not is_binary:
            try:
                # concatenate list of (N_i, C) into (N, C)
                if 'all_val_probs_full' in locals() and all_val_probs_full:
                    y_true = np.asarray(all_val_labels).astype(int)
                    y_prob = np.concatenate(all_val_probs_full, axis=0)  # shape (N, C)
                    if y_prob.ndim == 2 and y_prob.shape[1] >= 3:
                        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            except Exception:
                roc_auc = None
                



      # ====== Val Saving, Plots & Prints =======

        plot_gradient_and_lr(current_epoch_gradient_norms, current_epoch_lrs, epoch, output_dir)
        plot_prob_distribution(all_train_probs, epoch, output_dir, is_binary)
        plot_loss(history, epoch, output_dir)
        plot_confusion_matrix(all_val_labels, all_val_preds, epoch, output_dir, is_binary)

        current_lr = float(optimizer.param_groups[0]['lr'])
        epoch_time = time.time() - epoch_start_time

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_score'].append(train_score)
        history['val_score'].append(val_score)
        history['epoch_time'].append(epoch_time)
        history['learning_rate'].append(current_lr)
        history['train_recall'].append(_to_scalar(train_recall))
        history['val_recall'].append(_to_scalar(val_recall))
        history['train_precision'].append(_to_scalar(train_precision))
        history['val_precision'].append(_to_scalar(val_precision))
        history['train_f1'].append(_to_scalar(train_f1))
        history['val_f1'].append(_to_scalar(val_f1))
        history['roc_auc'].append(roc_auc)

        print(
            f"Epoch {epoch + 1}| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Score: {_to_scalar(train_score):.4f} | Val Score: \033[1;33m{_to_scalar(val_score):.4f}\033[0m | "
            f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s | ",
            flush=True
        )
        print(
            f"\nEpoch {epoch + 1}| Train Precision: {_to_scalar(train_precision):.4f} | "
            f"Val Precision: \033[1m{_to_scalar(val_precision):.4f}\033[0m | "
            f"Train Recall: {_to_scalar(train_recall):.4f} | "
            f"Val Recall: \033[1m{_to_scalar(val_recall):.4f}\033[0m | "
            f"F1 Train: {_to_scalar(train_f1):.4f} | F1 Val: {_to_scalar(val_f1):.4f}"
            f" | ROC-AUC: {roc_auc if roc_auc is not None else float('nan'):.4f}",
            flush=True 
        )


        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            last_saved_epoch_loss = epoch + 1
            history['last_saved_epoch_loss'] = last_saved_epoch_loss

            save_path_loss = output_dir / f'Best_Model_loss.pth'   # {epoch + 1} - add to name to save every epoch
            save_dict_loss = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'metric': 'loss',
                'problem_type': problem_type
            }
            if scheduler is not None:
                save_dict_loss['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict_loss, save_path_loss)
            print(f"--> Saved as best model based on loss at epoch {epoch + 1}", flush=True)

        # Save best model (F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts_f1 = copy.deepcopy(model.state_dict())
            last_saved_epoch_f1 = epoch + 1
            history['last_saved_epoch_f1'] = last_saved_epoch_f1

            save_path_f1 = output_dir / f'Best_Model_f1.pth'
            save_dict_f1 = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_f1': best_val_f1,
                'metric': 'f1',
                'problem_type': problem_type
            }
            if scheduler is not None:
                save_dict_f1['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict_f1, save_path_f1)
            print(f"--> Saved as best model based on F1 score at epoch {epoch + 1}", flush=True)


    # Load best model weights
    use_f1 = (save_metric == 'f1' and best_val_f1 > float('-inf') and best_model_wts_f1 is not None)
    if use_f1:
        model.load_state_dict(best_model_wts_f1)
        print(f"Best model saved based on F1 score at epoch {last_saved_epoch_f1}.")
    else:
        model.load_state_dict(best_model_wts_loss)
        print(f"Best model saved based on loss at epoch {last_saved_epoch_loss}.")

    
    return model, history



