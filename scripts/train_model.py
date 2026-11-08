"""FaceAttend AI - PyTorch ArcFace Deep Metric Fine-Tuning Pipeline.

Implements MobileFaceNet backbone with Additive Angular Margin Loss (ArcFace),
training loop with Cosine Annealing, evaluation metrics, and ONNX export.
Designed for execution on Kaggle NVIDIA GPUs (T4/P100) or local CUDA/CPU.
"""

import argparse
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import config


# ============================================================================
# 1. ArcFace (Additive Angular Margin Loss) Head
# ============================================================================

class ArcMarginProduct(nn.Module):
    """Additive Angular Margin (ArcFace) classification head.

    Applies an angular penalty m to the ground-truth class angle theta:
    cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
    """

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 10,
        s: float = 64.0,       # Hypersphere scale radius
        m: float = 0.50,       # Angular margin penalty
        easy_margin: bool = False,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)

        # Weight matrix (out_features x in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 1. Normalize input features and class weight vectors
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))

        # 2. Compute cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. Apply angular margin penalty to ground truth class
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# ============================================================================
# 2. MobileFaceNet Lightweight Backbone
# ============================================================================

class ConvBlock(nn.Module):
    """Convolution + BatchNorm + PReLU block."""
    def __init__(self, in_c, out_c, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class ResidualBottleneck(nn.Module):
    """Inverted Residual Bottleneck with depthwise separable convolution."""
    def __init__(self, in_c, out_c, stride, expansion_factor=2):
        super(ResidualBottleneck, self).__init__()
        self.use_residual = (stride == 1 and in_c == out_c)
        hidden_dim = in_c * expansion_factor

        self.block = nn.Sequential(
            ConvBlock(in_c, hidden_dim, 1, 1, 0),
            ConvBlock(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileFaceNet(nn.Module):
    """Lightweight MobileFaceNet mapping 112x112 aligned face to 128-D embedding."""

    def __init__(self, embedding_size: int = 128):
        super(MobileFaceNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)  # 112 -> 56
        self.conv2 = ConvBlock(64, 64, 3, 1, 1, groups=64)

        self.bottlenecks = nn.Sequential(
            ResidualBottleneck(64, 64, 2),   # 56 -> 28
            ResidualBottleneck(64, 64, 1),
            ResidualBottleneck(64, 128, 2),  # 28 -> 14
            ResidualBottleneck(128, 128, 1),
            ResidualBottleneck(128, 128, 2), # 14 -> 7
            ResidualBottleneck(128, 128, 1),
        )

        self.conv3 = ConvBlock(128, 256, 1, 1, 0)
        self.gdc = nn.Conv2d(256, 256, 7, 1, 0, groups=256, bias=False)  # Global Depthwise Conv (7x7 -> 1x1)
        self.bn_gdc = nn.BatchNorm2d(256)

        self.linear = nn.Linear(256, embedding_size, bias=False)
        self.bn_linear = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bottlenecks(x)
        x = self.conv3(x)
        x = self.bn_gdc(self.gdc(x))
        x = torch.flatten(x, 1)
        x = self.bn_linear(self.linear(x))

        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


# ============================================================================
# 3. Demonstration & Benchmark Dataset Generator
# ============================================================================

class SyntheticFaceVerificationDataset(Dataset):
    """Generates synthetic multi-identity facial feature batches for verification and training.

    Ensures training can execute in any environment even before external datasets are downloaded.
    """

    def __init__(self, num_identities: int = 20, samples_per_identity: int = 25):
        self.num_identities = num_identities
        self.samples_per_identity = samples_per_identity
        self.total_samples = num_identities * samples_per_identity

        # Generate distinct canonical facial templates
        np.random.seed(42)
        self.templates = np.random.randn(num_identities, 3, 112, 112).astype(np.float32)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        identity_id = idx // self.samples_per_identity
        base_template = self.templates[identity_id]

        # Add random visual perturbation (lighting, micro-shift, noise)
        noise = np.random.randn(3, 112, 112).astype(np.float32) * 0.15
        image = np.clip(base_template + noise, -1.0, 1.0)

        tensor_img = torch.from_numpy(image)
        label = torch.tensor(identity_id, dtype=torch.long)
        return tensor_img, label


# ============================================================================
# 4. Training Engine
# ============================================================================

def train_arcface_model(
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 20,
    embedding_dim: int = 128,
    margin: float = 0.50,
    scale: float = 64.0,
    output_dir: Path = config.MODELS_DIR,
) -> dict:
    """Train the MobileFaceNet backbone with ArcFace loss and export to ONNX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"FaceAttend AI - Model Fine-Tuning Pipeline (Device: {device})")
    print(f"Backbone: MobileFaceNet | Head: ArcFace (s={scale}, m={margin})")
    print(f"Embedding Dim: {embedding_dim} | Classes: {num_classes} | Epochs: {epochs}")
    print("=" * 65)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Dataset & Loader
    dataset = SyntheticFaceVerificationDataset(num_identities=num_classes, samples_per_identity=30)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 2. Initialize Models
    backbone = MobileFaceNet(embedding_size=embedding_dim).to(device)
    arcface_head = ArcMarginProduct(in_features=embedding_dim, out_features=num_classes, s=scale, m=margin).to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(arcface_head.parameters()),
        lr=learning_rate,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Metrics history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    t_start = time.time()

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        backbone.train()
        arcface_head.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = backbone(images)
            outputs = arcface_head(embeddings, labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        # Validation Step
        backbone.eval()
        arcface_head.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = backbone(images)
                outputs = arcface_head(embeddings, labels)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_acc:5.1f}% | "
            f"Val Acc: {val_acc:5.1f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    elapsed = time.time() - t_start
    print(f"\n[*] Training completed in {elapsed:.1f}s ({elapsed / epochs:.2f}s/epoch).")

    # 5. Save Checkpoint (.pth)
    pth_path = output_dir / "custom_face_recognizer.pth"
    torch.save({
        "backbone_state": backbone.state_dict(),
        "arcface_state": arcface_head.state_dict(),
        "history": history,
        "embedding_dim": embedding_dim,
    }, pth_path)
    print(f"[*] Saved PyTorch weights to: {pth_path.name}")

    # 6. Plot Loss & Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(range(1, epochs + 1), history["train_loss"], "o-", color="#6366F1", lw=2)
    ax1.set_title("ArcFace Training Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy on Angular Margins")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(range(1, epochs + 1), history["train_acc"], "s-", color="#10B981", label="Train Accuracy", lw=2)
    ax2.plot(range(1, epochs + 1), history["val_acc"], "^--", color="#F59E0B", label="Validation Accuracy", lw=2)
    ax2.set_title("Biometric Verification Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    curve_path = output_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[*] Saved loss curves plot to: {curve_path.name}")

    # 7. Export to ONNX
    onnx_path = output_dir / "custom_face_recognizer.onnx"
    backbone.eval()
    dummy_input = torch.randn(1, 3, 112, 112, device=device)

    torch.onnx.export(
        backbone,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"[*] Exported ONNX inference model: {onnx_path.name} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 65)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcFace Feature Extractor")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_arcface_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
