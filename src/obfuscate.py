#!/usr/bin/env python3
"""
Obfuscation module - Obfuscate and minify HTML, CSS, and JavaScript.
"""

import re
import random
import string
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Comment
import cssutils
import logging

# Suppress cssutils warnings
cssutils.log.setLevel(logging.CRITICAL)

# Optional imports
try:
    import htmlmin
    HTMLMIN_AVAILABLE = True
except ImportError:
    HTMLMIN_AVAILABLE = False

try:
    from jsmin import jsmin
    JSMIN_AVAILABLE = True
except ImportError:
    JSMIN_AVAILABLE = False


class Obfuscator:
    """Obfuscate and minify web files."""

    def __init__(self, minify: bool = True, obfuscate: bool = True):
        self.minify = minify
        self.obfuscate = obfuscate
        self._class_map = {}
        self._id_map = {}

    def process_directory(self, base_path: Path, output_path: Optional[Path], excludes: list[str]) -> int:
        """Process all HTML, CSS, and JS files in directory."""
        count = 0

        for pattern in ['**/*.html', '**/*.htm']:
            for file_path in base_path.glob(pattern):
                if self._is_excluded(file_path, excludes):
                    continue
                if self._process_html(file_path, output_path):
                    count += 1

        for file_path in base_path.glob('**/*.css'):
            if self._is_excluded(file_path, excludes):
                continue
            if self._process_css(file_path, output_path):
                count += 1

        for file_path in base_path.glob('**/*.js'):
            if self._is_excluded(file_path, excludes):
                continue
            if self._process_js(file_path, output_path):
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

    def _generate_obfuscated_name(self, original: str, prefix: str = '') -> str:
        """Generate a short obfuscated name."""
        chars = string.ascii_lowercase
        length = 4 + len(original) % 3
        return prefix + ''.join(random.choices(chars, k=length))

    def _process_html(self, file_path: Path, output_path: Optional[Path]) -> bool:
        """Process an HTML file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(content, 'lxml')

            if self.obfuscate:
                # Remove HTML comments (but keep IE conditionals)
                for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
                    if not comment.strip().startswith('[if'):
                        comment.extract()

                # Obfuscate class names (map them consistently)
                for tag in soup.find_all(class_=True):
                    new_classes = []
                    for cls in tag.get('class', []):
                        if cls not in self._class_map:
                            self._class_map[cls] = self._generate_obfuscated_name(cls, 'c')
                        new_classes.append(self._class_map[cls])
                    tag['class'] = new_classes

                # Obfuscate IDs
                for tag in soup.find_all(id=True):
                    old_id = tag['id']
                    if old_id not in self._id_map:
                        self._id_map[old_id] = self._generate_obfuscated_name(old_id, 'i')
                    tag['id'] = self._id_map[old_id]

                # Update href="#id" references
                for tag in soup.find_all(href=re.compile(r'^#')):
                    old_ref = tag['href'][1:]
                    if old_ref in self._id_map:
                        tag['href'] = '#' + self._id_map[old_ref]

            result = str(soup)

            if self.minify and HTMLMIN_AVAILABLE:
                result = htmlmin.minify(
                    result,
                    remove_comments=True,
                    remove_empty_space=True,
                    reduce_boolean_attributes=True
                )

            out_file = self._get_output_path(file_path, output_path)
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
            out_file.write_text(result, encoding='utf-8')
            return True

        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")
            return False

    def _process_css(self, file_path: Path, output_path: Optional[Path]) -> bool:
        """Process a CSS file."""
        try:
            content = file_path.read_text(encoding='utf-8')

            if self.obfuscate:
                # Replace class names that we've mapped
                for old_cls, new_cls in self._class_map.items():
                    content = re.sub(rf'\.{re.escape(old_cls)}(?=\s|{{|,|:|\[|\.)', f'.{new_cls}', content)

                # Replace IDs
                for old_id, new_id in self._id_map.items():
                    content = re.sub(rf'#{re.escape(old_id)}(?=\s|{{|,|:|\[)', f'#{new_id}', content)

                # Remove CSS comments
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

            if self.minify:
                # Simple CSS minification
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'\s*([{};:,>+~])\s*', r'\1', content)
                content = content.strip()

            out_file = self._get_output_path(file_path, output_path)
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
            out_file.write_text(content, encoding='utf-8')
            return True

        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")
            return False

    def _process_js(self, file_path: Path, output_path: Optional[Path]) -> bool:
        """Process a JavaScript file."""
        try:
            content = file_path.read_text(encoding='utf-8')

            if self.obfuscate:
                # Update class references in JS (querySelector, classList, etc.)
                for old_cls, new_cls in self._class_map.items():
                    content = re.sub(rf"'\.{re.escape(old_cls)}'", f"'.{new_cls}'", content)
                    content = re.sub(rf'"\\.{re.escape(old_cls)}"', f'".{new_cls}"', content)
                    content = re.sub(rf"'{re.escape(old_cls)}'", f"'{new_cls}'", content)
                    content = re.sub(rf'"{re.escape(old_cls)}"', f'"{new_cls}"', content)

                # Update ID references
                for old_id, new_id in self._id_map.items():
                    content = re.sub(rf"'#{re.escape(old_id)}'", f"'#{new_id}'", content)
                    content = re.sub(rf'"#{re.escape(old_id)}"', f'"#{new_id}"', content)
                    content = re.sub(rf"getElementById\s*\(\s*'{re.escape(old_id)}'\s*\)", f"getElementById('{new_id}')", content)
                    content = re.sub(rf'getElementById\s*\(\s*"{re.escape(old_id)}"\s*\)', f'getElementById("{new_id}")', content)

            if self.minify and JSMIN_AVAILABLE:
                content = jsmin(content)

            out_file = self._get_output_path(file_path, output_path)
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
            out_file.write_text(content, encoding='utf-8')
            return True

        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")
            return False
