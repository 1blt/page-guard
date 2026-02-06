#!/usr/bin/env python3
"""
Image Shield - Protect images from AI training with adversarial perturbations.

Applies frequency-domain watermarking, wavelet perturbations, and strategic noise
to make images less useful for AI model training while remaining visually identical
to humans.
"""

import argparse
import glob
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dct, idct

# Optional: wavelet transforms
try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False

# Optional: neural network adversarial perturbations
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ProtectionConfig:
    """Configuration for image protection strength."""
    dct_strength: float
    wavelet_strength: float
    noise_strength: float
    adversarial_strength: float

    @classmethod
    def subtle(cls) -> 'ProtectionConfig':
        return cls(
            dct_strength=0.03,
            wavelet_strength=0.02,
            noise_strength=0.008,
            adversarial_strength=0.01
        )

    @classmethod
    def balanced(cls) -> 'ProtectionConfig':
        return cls(
            dct_strength=0.06,
            wavelet_strength=0.04,
            noise_strength=0.012,
            adversarial_strength=0.02
        )

    @classmethod
    def maximum(cls) -> 'ProtectionConfig':
        return cls(
            dct_strength=0.12,
            wavelet_strength=0.08,
            noise_strength=0.02,
            adversarial_strength=0.04
        )


class ImageProtector:
    """Applies multi-layer protection to images."""

    def __init__(self, config: ProtectionConfig, full_protection: bool = False):
        self.config = config
        self.full_protection = full_protection
        self.model = None

        if full_protection and TORCH_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load ResNet50 for adversarial perturbation generation."""
        print("Loading neural network model for full protection...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def protect(self, image: np.ndarray) -> np.ndarray:
        """Apply all protection layers to an image."""
        # Work in float32 for precision
        img = image.astype(np.float32) / 255.0

        # Layer 1: DCT watermarking (frequency domain)
        img = self._apply_dct_watermark(img)

        # Layer 2: Wavelet perturbation
        if WAVELET_AVAILABLE:
            img = self._apply_wavelet_perturbation(img)

        # Layer 3: Strategic noise injection
        img = self._apply_strategic_noise(img)

        # Layer 4: Adversarial perturbation (if enabled)
        if self.full_protection and self.model is not None:
            img = self._apply_adversarial_perturbation(img)

        # Clip and convert back to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def _apply_dct_watermark(self, img: np.ndarray) -> np.ndarray:
        """Apply DCT-based watermarking to mid-frequency components."""
        result = img.copy()

        for channel in range(min(3, img.shape[2]) if len(img.shape) > 2 else 1):
            if len(img.shape) > 2:
                plane = img[:, :, channel]
            else:
                plane = img

            # Apply 2D DCT
            dct_coeffs = dct(dct(plane.T, norm='ortho').T, norm='ortho')

            # Perturb mid-frequency coefficients
            h, w = dct_coeffs.shape
            mid_h, mid_w = h // 4, w // 4

            # Generate deterministic but image-specific noise
            np.random.seed(int(np.sum(plane[:10, :10]) * 1000) % (2**31))
            noise = np.random.randn(mid_h, mid_w) * self.config.dct_strength

            # Apply to mid-frequency band
            dct_coeffs[mid_h:mid_h*2, mid_w:mid_w*2] += noise

            # Inverse DCT
            reconstructed = idct(idct(dct_coeffs.T, norm='ortho').T, norm='ortho')

            if len(img.shape) > 2:
                result[:, :, channel] = reconstructed
            else:
                result = reconstructed

        return result

    def _apply_wavelet_perturbation(self, img: np.ndarray) -> np.ndarray:
        """Apply wavelet-domain perturbations."""
        result = img.copy()

        for channel in range(min(3, img.shape[2]) if len(img.shape) > 2 else 1):
            if len(img.shape) > 2:
                plane = img[:, :, channel]
            else:
                plane = img

            # 2-level wavelet decomposition
            coeffs = pywt.wavedec2(plane, 'db4', level=2)

            # Perturb detail coefficients
            modified_coeffs = [coeffs[0]]  # Keep approximation
            for level_coeffs in coeffs[1:]:
                modified_level = []
                for detail in level_coeffs:
                    np.random.seed(int(np.sum(detail[:5, :5].flatten()) * 1000) % (2**31))
                    noise = np.random.randn(*detail.shape) * self.config.wavelet_strength
                    modified_level.append(detail + noise)
                modified_coeffs.append(tuple(modified_level))

            # Reconstruct
            reconstructed = pywt.waverec2(modified_coeffs, 'db4')

            # Handle size mismatch from wavelet transform
            reconstructed = reconstructed[:plane.shape[0], :plane.shape[1]]

            if len(img.shape) > 2:
                result[:, :, channel] = reconstructed
            else:
                result = reconstructed

        return result

    def _apply_strategic_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply strategic noise that targets AI feature extraction."""
        # Combine Gaussian noise with edge-aware perturbation

        # Gaussian noise
        noise = np.random.randn(*img.shape) * self.config.noise_strength

        # Edge-aware: stronger noise near edges (where AI models focus)
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(img.shape) > 2 else (img * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        if len(img.shape) > 2:
            edges = np.stack([edges] * img.shape[2], axis=-1)

        # Apply stronger noise near edges
        edge_noise = np.random.randn(*img.shape) * self.config.noise_strength * 2
        combined_noise = noise + edges * edge_noise

        return img + combined_noise

    def _apply_adversarial_perturbation(self, img: np.ndarray) -> np.ndarray:
        """Apply neural network adversarial perturbations (FGSM-style)."""
        if not TORCH_AVAILABLE or self.model is None:
            return img

        # Prepare input tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert to PIL for transforms
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        input_tensor = transform(pil_img).unsqueeze(0)
        input_tensor.requires_grad = True

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # Forward pass
        output = self.model(input_tensor)

        # Get gradient w.r.t. input (target: confuse the model)
        target = output.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # FGSM perturbation
        perturbation = self.config.adversarial_strength * input_tensor.grad.sign()

        # Convert back to numpy
        perturbation = perturbation.squeeze().cpu().detach().numpy()
        perturbation = perturbation.transpose(1, 2, 0)  # CHW -> HWC

        # Resize perturbation to match image size
        if perturbation.shape[:2] != img.shape[:2]:
            perturbation = cv2.resize(perturbation, (img.shape[1], img.shape[0]))

        return img + perturbation


def find_images(base_path: str, formats: list[str], exclude_patterns: list[str]) -> list[Path]:
    """Find all images matching the specified formats, excluding patterns."""
    images = []
    base = Path(base_path)

    for fmt in formats:
        pattern = f"**/*.{fmt}"
        for img_path in base.glob(pattern):
            # Check exclusions
            excluded = False
            for excl in exclude_patterns:
                if img_path.match(excl):
                    excluded = True
                    break
            if not excluded:
                images.append(img_path)

    return images


def process_image(
    img_path: Path,
    output_path: Optional[Path],
    protector: ImageProtector,
    preserve_original: bool,
    base_path: Optional[Path] = None
) -> bool:
    """Process a single image. Returns True on success."""
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not read {img_path}")
            return False

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Protect
        protected = protector.protect(img)

        # Convert back to BGR for saving
        protected = cv2.cvtColor(protected, cv2.COLOR_RGB2BGR)

        # Determine output location
        if output_path:
            # Preserve directory structure relative to base_path
            if base_path:
                relative = img_path.relative_to(base_path)
            else:
                relative = Path(img_path.name)
            out_file = output_path / relative
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_file = img_path
            if preserve_original:
                original_backup = img_path.with_suffix(f'.original{img_path.suffix}')
                shutil.copy2(img_path, original_backup)

        # Save with appropriate quality
        ext = img_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(str(out_file), protected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif ext == '.png':
            cv2.imwrite(str(out_file), protected, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        elif ext == '.webp':
            cv2.imwrite(str(out_file), protected, [cv2.IMWRITE_WEBP_QUALITY, 95])
        else:
            cv2.imwrite(str(out_file), protected)

        return True
    except Exception as e:
        print(f"  Error processing {img_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Protect images from AI training')
    parser.add_argument('--images-path', default='.', help='Path to images directory')
    parser.add_argument('--output-path', default='', help='Output directory (empty = in-place)')
    parser.add_argument('--strength', choices=['subtle', 'balanced', 'maximum'], default='balanced')
    parser.add_argument('--formats', default='png,jpg,jpeg,webp,gif', help='Comma-separated formats')
    parser.add_argument('--exclude', default='node_modules/**,vendor/**,.git/**', help='Exclude patterns')
    parser.add_argument('--full-protection', action='store_true', help='Enable adversarial perturbations')
    parser.add_argument('--preserve-originals', action='store_true', help='Keep original files')

    args = parser.parse_args()

    # Parse config
    config_map = {
        'subtle': ProtectionConfig.subtle,
        'balanced': ProtectionConfig.balanced,
        'maximum': ProtectionConfig.maximum
    }
    config = config_map[args.strength]()

    formats = [f.strip().lstrip('.') for f in args.formats.split(',')]
    excludes = [e.strip() for e in args.exclude.split(',')]

    output_path = Path(args.output_path) if args.output_path else None

    # Check for full protection requirements
    if args.full_protection and not TORCH_AVAILABLE:
        print("Warning: --full-protection requires PyTorch. Falling back to standard protection.")
        args.full_protection = False

    if not WAVELET_AVAILABLE:
        print("Note: PyWavelets not available. Wavelet protection disabled.")

    # Find images
    print(f"Scanning {args.images_path} for images...")
    images = find_images(args.images_path, formats, excludes)

    if not images:
        print("No images found.")
        print("::set-output name=processed::0")
        print("::set-output name=failed::0")
        return 0

    print(f"Found {len(images)} images to protect")

    # Initialize protector
    protector = ImageProtector(config, full_protection=args.full_protection)

    # Process images
    processed = 0
    failed = 0

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing {img_path.name}...")
        if process_image(img_path, output_path, protector, args.preserve_originals):
            processed += 1
        else:
            failed += 1

    # Output for GitHub Actions
    print(f"\nComplete: {processed} protected, {failed} failed")

    # GitHub Actions outputs
    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"processed={processed}\n")
            f.write(f"failed={failed}\n")
    else:
        # Fallback for local testing
        print(f"::set-output name=processed::{processed}")
        print(f"::set-output name=failed::{failed}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
