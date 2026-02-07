#!/usr/bin/env python3
"""
Text Guard module - Text transformation to prevent copy/search.

Transforms visible text content to break text search and copy operations
while maintaining visual appearance:
- Homoglyph substitution: Replace characters with visually identical Unicode lookalikes
- Zero-width character insertion: Insert invisible characters between letters

These transformations only affect text nodes, not:
- Element IDs, classes, or attributes
- JavaScript code
- CSS content
- Framework bindings
"""

import random
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString

# Homoglyph mappings: ASCII -> visually similar Unicode characters
# These look identical in most fonts but have different code points
HOMOGLYPHS = {
    'a': '\u0430',  # Cyrillic small letter a
    'c': '\u0441',  # Cyrillic small letter es
    'e': '\u0435',  # Cyrillic small letter ie
    'o': '\u043e',  # Cyrillic small letter o
    'p': '\u0440',  # Cyrillic small letter er
    'x': '\u0445',  # Cyrillic small letter ha
    'y': '\u0443',  # Cyrillic small letter u
    's': '\u0455',  # Cyrillic small letter dze
    'i': '\u0456',  # Cyrillic small letter byelorussian-ukrainian i
    'j': '\u0458',  # Cyrillic small letter je
    'A': '\u0410',  # Cyrillic capital letter A
    'B': '\u0412',  # Cyrillic capital letter Ve
    'C': '\u0421',  # Cyrillic capital letter Es
    'E': '\u0415',  # Cyrillic capital letter Ie
    'H': '\u041d',  # Cyrillic capital letter En
    'K': '\u041a',  # Cyrillic capital letter Ka
    'M': '\u041c',  # Cyrillic capital letter Em
    'O': '\u041e',  # Cyrillic capital letter O
    'P': '\u0420',  # Cyrillic capital letter Er
    'T': '\u0422',  # Cyrillic capital letter Te
    'X': '\u0425',  # Cyrillic capital letter Ha
    'Y': '\u0423',  # Cyrillic capital letter U
}

# Zero-width characters that are invisible but break text matching
ZERO_WIDTH_CHARS = [
    '\u200b',  # Zero-width space
    '\u200c',  # Zero-width non-joiner
    '\u200d',  # Zero-width joiner
    '\ufeff',  # Zero-width no-break space (BOM)
]

# Tags whose content should never be transformed
SKIP_TAGS = frozenset([
    'script', 'style', 'code', 'pre', 'textarea',
    'input', 'select', 'option', 'noscript', 'template',
    'svg', 'math', 'head', 'title', 'meta', 'link',
])

# Attributes that mark elements to skip
SKIP_ATTRIBUTES = frozenset([
    'data-no-transform',
    'data-textguard-skip',
    'contenteditable',
])


class TextGuard:
    """Transform text content to prevent copy/search operations."""

    def __init__(
        self,
        homoglyphs: bool = True,
        zero_width: bool = True,
        homoglyph_ratio: float = 0.3,
        zero_width_ratio: float = 0.2,
    ):
        """Initialize TextGuard.

        Args:
            homoglyphs: Enable homoglyph substitution
            zero_width: Enable zero-width character insertion
            homoglyph_ratio: Fraction of eligible characters to replace (0.0-1.0)
            zero_width_ratio: Fraction of character gaps to fill with zero-width chars (0.0-1.0)
        """
        self.homoglyphs = homoglyphs
        self.zero_width = zero_width
        self.homoglyph_ratio = max(0.0, min(1.0, homoglyph_ratio))
        self.zero_width_ratio = max(0.0, min(1.0, zero_width_ratio))
        self._base_path: Optional[Path] = None
        self._chars_transformed = 0
        self._zero_width_inserted = 0

    def process_directory(self, base_path: Path, output_path: Optional[Path], excludes: list[str]) -> dict:
        """Process all HTML files in directory.

        Returns:
            dict with 'chars_transformed' and 'zero_width_inserted' counts
        """
        self._base_path = base_path
        self._chars_transformed = 0
        self._zero_width_inserted = 0

        for pattern in ['**/*.html', '**/*.htm']:
            for file_path in base_path.glob(pattern):
                if self._is_excluded(file_path, excludes):
                    continue
                self._process_html(file_path, output_path)

        return {
            'chars_transformed': self._chars_transformed,
            'zero_width_inserted': self._zero_width_inserted,
        }

    def _is_excluded(self, path: Path, excludes: list[str]) -> bool:
        """Check if path matches any exclude pattern."""
        for excl in excludes:
            if path.match(excl):
                return True
        return False

    def _get_output_path(self, file_path: Path, output_path: Optional[Path]) -> Path:
        """Get the output path for a file, preserving directory structure."""
        if output_path:
            if self._base_path:
                relative = file_path.relative_to(self._base_path)
            else:
                relative = Path(file_path.name)
            return output_path / relative
        return file_path

    def _should_skip_element(self, element) -> bool:
        """Check if an element or its ancestors should be skipped."""
        current = element
        while current:
            if hasattr(current, 'name'):
                # Check tag name
                if current.name and current.name.lower() in SKIP_TAGS:
                    return True
                # Check attributes
                if hasattr(current, 'attrs'):
                    for attr in SKIP_ATTRIBUTES:
                        if attr in current.attrs:
                            return True
            current = current.parent
        return False

    def _transform_text(self, text: str) -> str:
        """Apply transformations to a text string."""
        if not text or not text.strip():
            return text

        result = []
        for i, char in enumerate(text):
            transformed_char = char

            # Apply homoglyph substitution
            if self.homoglyphs and char in HOMOGLYPHS:
                if random.random() < self.homoglyph_ratio:
                    transformed_char = HOMOGLYPHS[char]
                    self._chars_transformed += 1

            result.append(transformed_char)

            # Insert zero-width character after this character (but not at end)
            if self.zero_width and i < len(text) - 1:
                if char.isalnum() and random.random() < self.zero_width_ratio:
                    result.append(random.choice(ZERO_WIDTH_CHARS))
                    self._zero_width_inserted += 1

        return ''.join(result)

    def _process_html(self, file_path: Path, output_path: Optional[Path]) -> bool:
        """Process an HTML file, transforming text nodes."""
        try:
            content = file_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(content, 'lxml')

            # Find all text nodes and transform them
            for text_node in soup.find_all(string=True):
                # Skip if not a NavigableString (could be Comment, etc.)
                if not isinstance(text_node, NavigableString):
                    continue

                # Skip if in a protected element
                if self._should_skip_element(text_node):
                    continue

                # Skip whitespace-only nodes
                if not text_node.strip():
                    continue

                # Transform the text
                original = str(text_node)
                transformed = self._transform_text(original)

                if transformed != original:
                    text_node.replace_with(NavigableString(transformed))

            # Write output
            out_file = self._get_output_path(file_path, output_path)
            if output_path:
                out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(str(soup), encoding='utf-8')
            return True

        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")
            return False
