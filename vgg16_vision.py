"""
VGG16 视觉模块：基于预训练 VGG16 的图像分类与特征提取。

依赖: torch, torchvision, pillow, numpy
可选: opencv-python (用于读取本地图片或视频帧)
"""

import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG16_Weights


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_device(device: Optional[str] = None) -> torch.device:
    """自动选择设备；CPU 回退时给出原因提示。"""
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] 使用 GPU: {name}")
        return torch.device("cuda")

    print(f"[INFO] 使用设备: cpu")
    if torch.version.cuda is None:
        print(
            "[WARNING] 当前 PyTorch 为 CPU 版 "
            f"({torch.__version__})，未编译 CUDA 支持。"
            "如需 GPU 加速，请重装 CUDA 版 PyTorch，例如:\n"
            "  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu126"
        )
    else:
        print("[WARNING] PyTorch 含 CUDA，但未检测到可用 GPU，请检查 NVIDIA 驱动。")
    return torch.device("cpu")


class VGG16Vision:
    """VGG16 视觉推理封装：分类、特征提取。"""

    def __init__(self, device: Optional[str] = None, pretrained: bool = True):
        """
        Args:
            device: 计算设备，如 'cuda' 或 'cpu'；None 时自动选择。
            pretrained: 是否加载 ImageNet 预训练权重。
        """
        if device is None:
            self.device = resolve_device()
        else:
            self.device = torch.device(device)

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vgg16(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        self.weights_meta = VGG16_Weights.IMAGENET1K_V1
        self.categories = self.weights_meta.meta["categories"]
        self.preprocess = self.weights_meta.transforms()

    def _to_pil(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """将路径、PIL 图像或 OpenCV BGR 数组转为 RGB PIL 图像。"""
        if isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"图像不存在: {image}")
            return Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                arr = image
            elif image.ndim == 3 and image.shape[2] == 3:
                # OpenCV 默认 BGR
                arr = image[:, :, ::-1]
            elif image.ndim == 3 and image.shape[2] == 4:
                arr = image[:, :, :3][:, :, ::-1]
            else:
                raise ValueError(f"不支持的数组形状: {image.shape}")
            return Image.fromarray(arr.astype(np.uint8))

        raise TypeError(f"不支持的图像类型: {type(image)}")

    def _prepare_tensor(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        pil_img = self._to_pil(image)
        tensor = self.preprocess(pil_img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def classify(
        self,
        image: Union[str, np.ndarray, Image.Image],
        top_k: int = 5,
    ) -> list[dict]:
        """
        对单张图像做 ImageNet 分类。

        Returns:
            [{"label": str, "confidence": float}, ...]
        """
        x = self._prepare_tensor(image)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(top_k, probs.numel())
        confidences, indices = torch.topk(probs, k)

        results = []
        for score, idx in zip(confidences.tolist(), indices.tolist()):
            results.append(
                {
                    "label": self.categories[idx],
                    "confidence": round(score, 4),
                }
            )
        return results

    @torch.inference_mode()
    def extract_features(
        self,
        image: Union[str, np.ndarray, Image.Image],
        layer: str = "avgpool",
    ) -> np.ndarray:
        """
        提取中间层特征向量。

        Args:
            layer: 'conv5_3' | 'avgpool' | 'fc7'
                - conv5_3: 卷积末层特征图 (512, 7, 7)
                - avgpool: 全局平均池化后 (512,)
                - fc7: 全连接倒数第二层 (4096,)
        """
        x = self._prepare_tensor(image)
        features = {}

        def hook(name):
            def _hook(_module, _input, output):
                features[name] = output.detach()

            return _hook

        handles = []
        if layer == "conv5_3":
            handles.append(self.model.features[28].register_forward_hook(hook("out")))
        elif layer in ("avgpool", "fc7"):
            handles.append(self.model.avgpool.register_forward_hook(hook("out")))
        else:
            raise ValueError(f"不支持的 layer: {layer}")

        _ = self.model(x)

        for h in handles:
            h.remove()

        feat = features["out"]
        if layer == "fc7":
            feat = torch.flatten(self.model.classifier[:4](feat), 1)

        return feat.squeeze(0).cpu().numpy()

    def build_feature_extractor(self, layer: str = "avgpool") -> nn.Module:
        """
        构建截断后的 VGG16，用于批量特征提取。

        Args:
            layer: 'conv5_3' | 'avgpool' | 'fc7'
        """
        if layer == "conv5_3":
            return nn.Sequential(self.model.features).eval().to(self.device)

        if layer == "avgpool":
            return nn.Sequential(
                self.model.features,
                self.model.avgpool,
                nn.Flatten(),
            ).eval().to(self.device)

        if layer == "fc7":
            return nn.Sequential(
                self.model.features,
                self.model.avgpool,
                nn.Flatten(),
                self.model.classifier[:4],
            ).eval().to(self.device)

        raise ValueError(f"不支持的 layer: {layer}")


def print_classification_results(results: list[dict]) -> None:
    """打印分类结果。"""
    print("[INFO] VGG16 分类结果:")
    for i, item in enumerate(results, 1):
        print(f"  {i}. {item['label']} ({item['confidence']:.2%})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VGG16 视觉推理示例")
    parser.add_argument(
        "image",
        nargs="?",
        default="",
        help="输入图像路径；留空则使用随机噪声图做演示",
    )
    parser.add_argument("--top-k", type=int, default=5, help="输出前 K 个类别")
    parser.add_argument(
        "--layer",
        type=str,
        default="avgpool",
        choices=["conv5_3", "avgpool", "fc7"],
        help="特征提取层",
    )
    args = parser.parse_args()

    print(f"[INFO] PyTorch 版本: {torch.__version__}")
    vision = VGG16Vision()

    if args.image:
        input_image = args.image
        print(f"[INFO] 读取图像: {args.image}")
    else:
        print("[INFO] 未指定图像，使用随机 RGB 图像演示")
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    results = vision.classify(input_image, top_k=args.top_k)
    print_classification_results(results)

    feat = vision.extract_features(input_image, layer=args.layer)
    print(f"[INFO] 特征层 {args.layer} 形状: {feat.shape}")
