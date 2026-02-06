#!/usr/bin/env python3
"""
Obfuscation module - Safe minification for HTML, CSS, and JavaScript.

This module performs only safe transformations that maintain functional equivalence:
- Comment removal
- Whitespace minification
- Safe formatting changes

It does NOT perform class/ID renaming as this breaks modern frameworks
(lit-html, React, Vue) that use dynamic class bindings.
"""

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Comment
import logging

# Suppress cssutils warnings
import cssutils
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
    """Safe minification for web files.

    Performs only transformations that preserve functional equivalence:
    - HTML: Comment removal, whitespace minification
    - CSS: Comment removal, whitespace minification
    - JS: Comment removal, whitespace minification

    Class and ID names are NOT modified as this breaks:
    - lit-html classMap() directive
    - React className bindings
    - Vue :class bindings
    - Any framework using dynamic class composition
    """

    def __init__(self, minify: bool = True, obfuscate: bool = True):
        """Initialize the obfuscator.

        Args:
            minify: Enable whitespace minification
            obfuscate: Enable comment removal (class/ID renaming is disabled for safety)
        """
        self.minify = minify
        self.obfuscate = obfuscate

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

    def _process_html(self, file_path: Path, output_path: Optional[Path]) -> bool:
        """Process an HTML file with safe transformations only."""
        try:
            content = file_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(content, 'lxml')

            if self.obfuscate:
                # Remove HTML comments (but keep IE conditionals)
                for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
                    if not comment.strip().startswith('[if'):
                        comment.extract()

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
        """Process a CSS file with safe transformations only."""
        try:
            content = file_path.read_text(encoding='utf-8')

            if self.obfuscate:
                # Remove CSS comments only - do not rename classes/IDs
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

            if self.minify:
                # Safe CSS minification - only whitespace changes
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
        """Process a JavaScript file with safe transformations only."""
        try:
            content = file_path.read_text(encoding='utf-8')

            # jsmin handles comment removal and safe minification
            # It does NOT rename variables or modify string contents
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
