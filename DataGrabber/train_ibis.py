#!/usr/bin/env python3
"""
Step 6 (v2) — train a White Ibis / not-ibis classifier and export to Core ML.

Changes from v1, all aimed at the overfitting we saw (train acc 0.92 / val 0.75):
  * Fine-tune only the LAST few backbone blocks, not the whole network
  * Added weight decay + extra dropout (regularization)
  * Track the single best model across BOTH phases (don't discard a better head)
  * Early stopping so a phase stops once val accuracy quits improving

Setup:
    conda activate ibis
    python train_ibis.py
"""

import torch
import torch.nn as nn
import coremltools as ct
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader

# ------------------------------- CONFIG -------------------------------
DATA_DIR = "data_split"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 8           # max; early stopping usually ends sooner
EPOCHS_FINETUNE = 20
PATIENCE = 5              # stop a phase after this many epochs with no val gain
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
WEIGHT_DECAY = 1e-4       # regularization: penalizes memorizing
UNFREEZE_LAST_N = 3       # how many trailing backbone blocks to fine-tune
DROPOUT = 0.4             # up from the default 0.2, for more regularization
OUT_COREML = "WildlifeClassifier.mlpackage"
OUT_WEIGHTS = "ibis_model.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# ----------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=eval_tf)
test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform=eval_tf)

class_names = train_ds.classes
target_idx = class_names.index("target")
print(f"Classes (by index): {class_names}")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4)

counts = [0] * len(class_names)
for _, y in train_ds.samples:
    counts[y] += 1
weights = torch.tensor([sum(counts) / (len(counts) * c) for c in counts], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
print(f"Train images per class: {dict(zip(class_names, counts))}")

model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
in_feats = model.classifier[-1].in_features
if isinstance(model.classifier[2], nn.Dropout):
    model.classifier[2].p = DROPOUT                       # stronger dropout
model.classifier[-1] = nn.Linear(in_feats, len(class_names))
model = model.to(device)

# single best across BOTH phases
best_val = 0.0
best_state = None


def run_epoch(loader, train=False, optimizer=None):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def train_phase(name, epochs, optimizer):
    global best_val, best_state
    stale = 0
    for ep in range(1, epochs + 1):
        _, tr_acc = run_epoch(train_dl, train=True, optimizer=optimizer)
        _, va_acc = run_epoch(val_dl, train=False)
        gap = tr_acc - va_acc
        print(f"[{name}] epoch {ep}/{epochs}  train_acc={tr_acc:.3f}  "
              f"val_acc={va_acc:.3f}  gap={gap:+.3f}")
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[{name}] early stop — no val gain in {PATIENCE} epochs")
                break


# Phase 1: freeze backbone, train only the new head
for p in model.features.parameters():
    p.requires_grad = False
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
train_phase("head", EPOCHS_HEAD, opt)

# Phase 2: unfreeze ONLY the last few blocks + head, fine-tune gently
for p in model.parameters():
    p.requires_grad = False
for p in model.features[-UNFREEZE_LAST_N:].parameters():
    p.requires_grad = True
for p in model.classifier.parameters():
    p.requires_grad = True
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
train_phase("finetune", EPOCHS_FINETUNE, opt)

# restore the best model seen in EITHER phase
if best_state:
    model.load_state_dict(best_state)
print(f"\nBest val_acc across all training: {best_val:.3f}")

# ---- honest evaluation on the held-out test set (step 7) ----
model.eval()
tp = fp = fn = tn = 0
with torch.no_grad():
    for x, y in test_dl:
        preds = model(x.to(device)).argmax(1).cpu()
        for pred, true in zip(preds, y):
            pred_ibis = (pred.item() == target_idx)
            true_ibis = (true.item() == target_idx)
            if pred_ibis and true_ibis:       tp += 1
            elif pred_ibis and not true_ibis:  fp += 1
            elif not pred_ibis and true_ibis:  fn += 1
            else:                              tn += 1

total = tp + fp + fn + tn
acc = (tp + tn) / total if total else 0
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
print("\n================ TEST RESULTS ================")
print(f"  accuracy : {acc:.3f}  ({tp + tn}/{total})")
print(f"  precision: {prec:.3f}   (of badge PASSES, how many were really ibis)")
print(f"  recall   : {rec:.3f}   (of real ibis, how many we caught)")
print(f"  false positives (NOT-ibis wrongly passed): {fp}   <- badge cheating risk")
print(f"  false negatives (real ibis wrongly failed): {fn}   <- annoyed-user risk")
print("==============================================\n")

torch.save(model.state_dict(), OUT_WEIGHTS)
print(f"Saved PyTorch weights -> {OUT_WEIGHTS}")

# ---- export to Core ML (normalization + softmax baked in) ----
class CoreMLWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, x):                     # x: RGB, 0-255, (1,3,H,W)
        x = (x - self.mean) / self.std
        return torch.softmax(self.net(x), dim=1)


wrapper = CoreMLWrapper(model.cpu()).eval()
example = torch.rand(1, 3, IMG_SIZE, IMG_SIZE) * 255.0
traced = torch.jit.trace(wrapper, example)
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, IMG_SIZE, IMG_SIZE),
                         color_layout=ct.colorlayout.RGB)],
    classifier_config=ct.ClassifierConfig(class_labels=class_names),
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.short_description = "White Ibis presence classifier"
mlmodel.save(OUT_COREML)
print(f"Saved Core ML model -> {OUT_COREML}")
