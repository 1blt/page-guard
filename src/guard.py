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

    args = parser.parse_args()

    excludes = [e.strip() for e in args.exclude.split(',')]
    output_path = Path(args.output_path) if args.output_path else None
    base_path = Path(args.path)

    # Stats
    stats = {
        'images_processed': 0,
        'files_obfuscated': 0,
        'metadata_stripped': 0,
        'decoys_injected': 0
    }

    # Track which protections are enabled
    protections = {
        'images': args.protect_images,
        'obfuscation': args.obfuscate or args.minify,
        'metadata': args.strip_metadata,
        'anti_scrape': args.anti_scrape or args.poison_data
    }

    print(f"Page Guard - Processing {base_path}")
    print("=" * 50)

    # 1. Image Protection
    if args.protect_images:
        print("\n[1/4] Image Protection")
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
        print("\n[1/4] Image Protection: Skipped")

    # 2. Obfuscation
    if args.obfuscate or args.minify:
        print("\n[2/4] Obfuscation")
        obfuscator = Obfuscator(minify=args.minify, obfuscate=args.obfuscate)
        count = obfuscator.process_directory(base_path, output_path, excludes)
        stats['files_obfuscated'] = count
        print(f"  Processed {count} files")
    else:
        print("\n[2/4] Obfuscation: Skipped")

    # 3. Metadata Stripping
    if args.strip_metadata:
        print("\n[3/4] Metadata Stripping")
        stripper = MetadataStripper()
        count = stripper.process_directory(base_path, output_path, excludes)
        stats['metadata_stripped'] = count
        print(f"  Stripped metadata from {count} files")
    else:
        print("\n[3/4] Metadata Stripping: Skipped")

    # 4. Anti-Scraping
    if args.anti_scrape or args.poison_data:
        print("\n[4/4] Anti-Scraping")
        scraper = AntiScraper(honeypots=args.anti_scrape, poison=args.poison_data)
        count = scraper.process_directory(base_path, output_path, excludes)
        stats['decoys_injected'] = count
        print(f"  Injected {count} decoy elements")
    else:
        print("\n[4/4] Anti-Scraping: Skipped")

    # Output stats for GitHub Actions
    print("\n" + "=" * 50)
    print("Complete!")

    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            for key, value in stats.items():
                f.write(f"{key}={value}\n")
    else:
        for key, value in stats.items():
            print(f"::set-output name={key}::{value}")

    # Generate GitHub Actions summary table
    github_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if github_summary:
        with open(github_summary, 'a') as f:
            f.write("## Page Guard Summary\n\n")

            # Protection status table
            f.write("### Protection Status\n\n")
            f.write("| Protection | Status | Count |\n")
            f.write("|------------|--------|-------|\n")

            def status_cell(enabled):
                return "Enabled" if enabled else "Disabled"

            f.write(f"| Image Protection | {status_cell(protections['images'])} | {stats['images_processed']} images |\n")
            f.write(f"| Obfuscation | {status_cell(protections['obfuscation'])} | {stats['files_obfuscated']} files |\n")
            f.write(f"| Metadata Stripping | {status_cell(protections['metadata'])} | {stats['metadata_stripped']} files |\n")
            f.write(f"| Anti-Scraping | {status_cell(protections['anti_scrape'])} | {stats['decoys_injected']} decoys |\n")
            f.write("\n")

            # Configuration details
            f.write("### Configuration\n\n")
            f.write(f"- **Path:** `{base_path}`\n")
            if output_path:
                f.write(f"- **Output:** `{output_path}`\n")
            else:
                f.write("- **Output:** In-place modification\n")
            if protections['images']:
                f.write(f"- **Image Strength:** {args.image_strength}\n")
                if args.full_image_protection:
                    f.write("- **Full Image Protection:** Enabled (neural network perturbations)\n")
            f.write("\n")

            # Total changes
            total = sum(stats.values())
            f.write(f"**Total changes:** {total}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
