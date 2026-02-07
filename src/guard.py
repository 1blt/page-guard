#!/usr/bin/env python3
"""
Page Guard - Protect your site from AI scraping.

Combines multiple protection layers:
- Image protection (adversarial perturbations)
- HTML/CSS/JS obfuscation and minification
- Metadata stripping
- Anti-scraping measures with decoy content
"""

import argparse
import os
import sys
from pathlib import Path

from protect import ImageProtector, ProtectionConfig, find_images, process_image
from obfuscate import Obfuscator
from metadata import MetadataStripper
from antiscrap import AntiScraper
from textguard import TextGuard


def main():
    parser = argparse.ArgumentParser(description='Page Guard - Protect your site from AI scraping')

    # General options
    parser.add_argument('--path', default='.', help='Path to site directory')
    parser.add_argument('--output-path', default='', help='Output directory (empty = in-place)')
    parser.add_argument('--exclude', default='node_modules/**,vendor/**,.git/**', help='Exclude patterns')

    # Image protection
    parser.add_argument('--protect-images', action='store_true', help='Enable image protection')
    parser.add_argument('--image-strength', choices=['subtle', 'balanced', 'maximum'], default='balanced')
    parser.add_argument('--image-formats', default='png,jpg,jpeg,webp', help='Image formats')
    parser.add_argument('--full-image-protection', action='store_true', help='Enable neural network perturbations')

    # Obfuscation
    parser.add_argument('--obfuscate', action='store_true', help='Enable obfuscation')
    parser.add_argument('--minify', action='store_true', help='Enable minification')

    # Metadata
    parser.add_argument('--strip-metadata', action='store_true', help='Strip metadata')

    # Anti-scraping
    parser.add_argument('--anti-scrape', action='store_true', help='Enable anti-scraping')
    parser.add_argument('--poison-data', action='store_true', help='Inject decoy content')

    # Text Guard
    parser.add_argument('--text-guard', action='store_true', help='Enable text transformation')
    parser.add_argument('--homoglyphs', action='store_true', help='Replace chars with lookalikes')
    parser.add_argument('--zero-width', action='store_true', help='Insert zero-width characters')
    parser.add_argument('--homoglyph-ratio', type=float, default=0.3, help='Homoglyph replacement ratio (0.0-1.0)')
    parser.add_argument('--zero-width-ratio', type=float, default=0.2, help='Zero-width insertion ratio (0.0-1.0)')

    # Output
    parser.add_argument('--no-summary', action='store_true', help='Disable GitHub Actions summary')

    args = parser.parse_args()

    excludes = [e.strip() for e in args.exclude.split(',')]
    output_path = Path(args.output_path) if args.output_path else None
    base_path = Path(args.path)

    # Stats
    stats = {
        'images_processed': 0,
        'files_obfuscated': 0,
        'metadata_stripped': 0,
        'decoys_injected': 0,
        'chars_transformed': 0,
        'zero_width_inserted': 0,
    }

    # Track which protections are enabled
    text_guard_enabled = args.text_guard or args.homoglyphs or args.zero_width
    protections = {
        'images': args.protect_images,
        'obfuscation': args.obfuscate or args.minify,
        'metadata': args.strip_metadata,
        'anti_scrape': args.anti_scrape or args.poison_data,
        'text_guard': text_guard_enabled,
    }

    print(f"Page Guard - Processing {base_path}")
    print("=" * 50)

    # 1. Metadata Stripping (first - clean images before protection)
    if args.strip_metadata:
        print("\n[1/5] Metadata Stripping")
        stripper = MetadataStripper()
        count = stripper.process_directory(base_path, output_path, excludes)
        stats['metadata_stripped'] = count
        print(f"  Stripped metadata from {count} files")
    else:
        print("\n[1/5] Metadata Stripping: Skipped")

    # 2. Image Protection (after metadata stripped, so perturbations persist)
    if args.protect_images:
        print("\n[2/5] Image Protection")
        config_map = {
            'subtle': ProtectionConfig.subtle,
            'balanced': ProtectionConfig.balanced,
            'maximum': ProtectionConfig.maximum
        }
        config = config_map[args.image_strength]()
        formats = [f.strip().lstrip('.') for f in args.image_formats.split(',')]

        images = find_images(str(base_path), formats, excludes)
        if images:
            protector = ImageProtector(config, full_protection=args.full_image_protection)
            for img_path in images:
                if process_image(img_path, output_path, protector, False, base_path):
                    stats['images_processed'] += 1
            print(f"  Protected {stats['images_processed']} images")
        else:
            print("  No images found")
    else:
        print("\n[2/5] Image Protection: Skipped")

    # 3. Obfuscation (minify HTML/CSS/JS)
    if args.obfuscate or args.minify:
        print("\n[3/5] Obfuscation")
        obfuscator = Obfuscator(minify=args.minify, obfuscate=args.obfuscate)
        count = obfuscator.process_directory(base_path, output_path, excludes)
        stats['files_obfuscated'] = count
        print(f"  Processed {count} files")
    else:
        print("\n[3/5] Obfuscation: Skipped")

    # 4. Text Guard (transform visible text to break copy/search)
    if text_guard_enabled:
        print("\n[4/5] Text Guard")
        use_homoglyphs = args.homoglyphs or args.text_guard
        use_zero_width = args.zero_width or args.text_guard
        guard = TextGuard(
            homoglyphs=use_homoglyphs,
            zero_width=use_zero_width,
            homoglyph_ratio=args.homoglyph_ratio,
            zero_width_ratio=args.zero_width_ratio,
        )
        result = guard.process_directory(base_path, output_path, excludes)
        stats['chars_transformed'] = result['chars_transformed']
        stats['zero_width_inserted'] = result['zero_width_inserted']
        print(f"  Transformed {result['chars_transformed']} chars, inserted {result['zero_width_inserted']} zero-width")
    else:
        print("\n[4/5] Text Guard: Skipped")

    # 5. Anti-Scraping (last - inject decoys into final HTML)
    if args.anti_scrape or args.poison_data:
        print("\n[5/5] Anti-Scraping")
        scraper = AntiScraper(honeypots=args.anti_scrape, poison=args.poison_data)
        count = scraper.process_directory(base_path, output_path, excludes)
        stats['decoys_injected'] = count
        print(f"  Injected {count} decoy elements")
    else:
        print("\n[5/5] Anti-Scraping: Skipped")

    # Output stats for GitHub Actions
    print("\n" + "=" * 50)
    print("Complete!")

    def status_label(enabled):
        return "Enabled" if enabled else "Disabled"

    total = sum(stats.values())

    # Write to GITHUB_OUTPUT for action outputs
    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            for key, value in stats.items():
                f.write(f"{key}={value}\n")

    # Generate summary (GitHub Actions or console)
    github_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    text_guard_count = f"{stats['chars_transformed']} chars, {stats['zero_width_inserted']} zero-width"
    if github_summary and not args.no_summary:
        with open(github_summary, 'a') as f:
            f.write("## Page Guard Summary\n\n")
            f.write("| Protection | Status | Count |\n")
            f.write("|------------|--------|-------|\n")
            f.write(f"| Image Protection | {status_label(protections['images'])} | {stats['images_processed']} images |\n")
            f.write(f"| Obfuscation | {status_label(protections['obfuscation'])} | {stats['files_obfuscated']} files |\n")
            f.write(f"| Metadata Stripping | {status_label(protections['metadata'])} | {stats['metadata_stripped']} files |\n")
            f.write(f"| Text Guard | {status_label(protections['text_guard'])} | {text_guard_count} |\n")
            f.write(f"| Anti-Scraping | {status_label(protections['anti_scrape'])} | {stats['decoys_injected']} decoys |\n")
            f.write(f"\n**Total changes:** {total}\n")
    else:
        # Console summary for local runs
        print(f"\nSummary:")
        print(f"  Images processed: {stats['images_processed']}")
        print(f"  Files obfuscated: {stats['files_obfuscated']}")
        print(f"  Metadata stripped: {stats['metadata_stripped']}")
        print(f"  Text guard: {text_guard_count}")
        print(f"  Decoys injected: {stats['decoys_injected']}")
        print(f"  Total: {total}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
