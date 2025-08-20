#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from pdf_extractor import PDFExtractor

def generate_single_output(input_file, output_file):
    """Processes a single PDF file and generates a JSON output."""
    pdf_path = Path(input_file)
    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {pdf_path.name}...")

    try:
        extractor = PDFExtractor(str(pdf_path))
        result = extractor.extract_data()
        extractor.close()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"✓ Generated {output_path.name}")
        print(f"  Title: {result['title']}")
        print(f"  Headings: {len(result['outline'])}")
        print()

    except Exception as e:
        print(f"✗ Error processing {pdf_path.name}: {e}")
        print()
        # Re-raise the exception to notify the calling process
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured data from a PDF file.")
    parser.add_argument("--input", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", required=True, help="Path to the output JSON file.")
    args = parser.parse_args()

    generate_single_output(args.input, args.output)
