#!/usr/bin/env python3
"""
Metadata stripping module - Remove EXIF, comments, and other metadata.
"""

import re
from pathlib import Path
from typing import Optional

from PIL import Image
from bs4 import BeautifulSoup, Comment

# Optional EXIF handling
try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False


class MetadataStripper:
    """Strip metadata from images and other files."""

    def process_directory(self, base_path: Path, output_path: Optional[Path], excludes: list[str]) -> int:
        """Process all files in directory."""
        count = 0

        # Process images
        for pattern in ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.webp', '**/*.tiff']:
            for file_path in base_path.glob(pattern):
                if self._is_excluded(file_path, excludes):
                    continue
                if self._strip_image_metadata(file_path, output_path):
                    count += 1

        # Process HTML (remove meta tags with sensitive info)
        for pattern in ['**/*.html', '**/*.htm']:
            for file_path in base_path.glob(pattern):
                if self._is_excluded(file_path, excludes):
                    continue
                if self._strip_html_metadata(file_path, output_path):
                    count += 1

        return count

    def _is_excluded(self, path: Path, excludes: list[str]) -> bool:
        """Check if path matches any exclude pattern."""
        for excl in excludes:
            if path.match(excl):
                return True
        return False

    def _get_output_path(self, file_path: Path, output_path: Optional[Path]) -> Path:
        """Get the output path for a file."""
        if output_path:
            return output_path / file_path.name
        return file_path

    def _strip_image_metadata(self, file_path: Path, output_path: Optional[Path]) -> bool:
        """Strip EXIF and other metadata from images."""
        try:
            img = Image.open(file_path)

            # Create clean copy without metadata
            data = list(img.getdata())
            clean_img = Image.new(img.mode, img.size)
            clean_img.putdata(data)

            out_file = self._get_output_path(file_path, output_path)
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)

            # Save with appropriate format
            ext = file_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                clean_img.save(out_file, 'JPEG', quality=95)
            elif ext == '.png':
                clean_img.save(out_file, 'PNG')
            elif ext == '.webp':
                clean_img.save(out_file, 'WEBP', quality=95)
            else:
                clean_img.save(out_file)

            return True

        except Exception as e:
            print(f"  Warning: Could not strip metadata from {file_path}: {e}")
            return False

    def _strip_html_metadata(self, file_path: Path, output_path: Optional[Path]) -> bool:
        """Remove sensitive meta tags and comments from HTML."""
        try:
            content = file_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(content, 'lxml')

            # Remove generator meta tags
            for meta in soup.find_all('meta', attrs={'name': re.compile(r'generator|author|created|modified', re.I)}):
                meta.decompose()

            # Remove HTML comments
            for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
                comment.extract()

            # Remove data-* attributes that might contain sensitive info
            sensitive_data_attrs = ['data-author', 'data-created', 'data-modified', 'data-user', 'data-email']
            for attr in sensitive_data_attrs:
                for tag in soup.find_all(attrs={attr: True}):
                    del tag[attr]

            out_file = self._get_output_path(file_path, output_path)
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
            out_file.write_text(str(soup), encoding='utf-8')
            return True

        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")
            return False
