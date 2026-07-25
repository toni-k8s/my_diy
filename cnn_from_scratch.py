"""
从零手搓 CNN（卷积神经网络）

用 PyTorch 实现一个最小可运行的 CNN，并在 MNIST 手写数字上训练演示。

CNN 核心思想：
  图像 → 卷积提取局部特征 → 池化降采样 → 多层堆叠 → 全连接分类
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """
    一个经典的两段式 CNN：

        输入 (1, 28, 28)
          ↓ Conv2d + ReLU      学局部模式（边缘、笔画）
          ↓ MaxPool2d          缩小尺寸、增强平移不变性
          ↓ Conv2d + ReLU      学更复杂的组合特征
          ↓ MaxPool2d
          ↓ Flatten            展平成向量
          ↓ Linear             映射到类别
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ---- 卷积块 1 ----
        # in_channels=1: MNIST 是灰度图
        # out_channels=16: 16 个不同的卷积核，每个核是一个「特征探测器」
        # kernel_size=3: 3×3 小窗口，每次只看局部区域
        # padding=1: 边缘补零，使 28×28 卷积后仍是 28×28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # MaxPool: 2×2 窗口取最大值，尺寸减半 (28→14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- 卷积块 2 ----
        # 16→32 通道：在上一层特征基础上学更抽象的模式
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # ---- 分类头 ----
        # 经过两次 pool: 28 → 14 → 7，所以空间尺寸是 7×7
        # 32 通道 × 7 × 7 = 1568 维
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        if verbose:
            print(f"  输入: {tuple(x.shape)}")

        # 卷积 = 滑动窗口做点积；ReLU 去掉负数，保留有效激活
        x = self.pool(F.relu(self.conv1(x)))
        if verbose:
            print(f"  conv1+relu+pool: {tuple(x.shape)}")

        x = self.pool(F.relu(self.conv2(x)))
        if verbose:
            print(f"  conv2+relu+pool: {tuple(x.shape)}")

        # 把 (batch, 32, 7, 7) 展平成 (batch, 1568)
        x = x.view(x.size(0), -1)
        if verbose:
            print(f"  flatten: {tuple(x.shape)}")

        # 全连接层输出每个类别的原始分数 (logits)
        x = self.fc(x)
        if verbose:
            print(f"  logits: {tuple(x.shape)}")

        return x


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向：算预测
        logits = model(images)
        # 交叉熵：衡量预测与真实标签的差距
        loss = F.cross_entropy(logits, labels)

        # 反向：算梯度
        optimizer.zero_grad()
        loss.backward()
        # 更新：按梯度调整权重
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return correct / len(loader.dataset)


def demo_forward_shapes(device: torch.device) -> None:
    """用一张假图片演示各层输出形状变化。"""
    print("\n[INFO] === 前向传播形状演示 ===")
    model = SimpleCNN().to(device)
    dummy = torch.randn(1, 1, 28, 28, device=device)
    model(dummy, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="手搓 CNN + MNIST 训练演示")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--demo-only", action="store_true", help="只演示前向形状，不训练")
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] 设备: {device}")

    demo_forward_shapes(device)

    if args.demo_only:
        return

    transform = transforms.Compose([
        transforms.ToTensor(),                          # PIL → (1,28,28)，值域 [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),     # MNIST 均值/方差标准化
    ])

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n[INFO] 开始训练 ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"[INFO] Epoch {epoch}/{args.epochs}  loss={loss:.4f}  test_acc={acc:.2%}")

    print("[INFO] 训练完成。")


if __name__ == "__main__":
    main()
