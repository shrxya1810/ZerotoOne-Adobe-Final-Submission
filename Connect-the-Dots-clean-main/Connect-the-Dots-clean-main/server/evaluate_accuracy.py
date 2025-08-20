import os
import json
import subprocess
import sys
from pathlib import Path

def compare_headings(actual_headings, expected_headings):
    """Compares two lists of heading objects and calculates precision, recall, and F1-score."""
    
    # Use a tuple of (text, level) for easy comparison
    actual_set = set((h['text'].strip(), h['level']) for h in actual_headings)
    expected_set = set((h['text'].strip(), h['level']) for h in expected_headings)

    true_positives = len(actual_set.intersection(expected_set))
    
    precision = true_positives / len(actual_set) if actual_set else 1.0
    recall = true_positives / len(expected_set) if expected_set else 1.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_headings": true_positives,
        "extra_headings": len(actual_set - expected_set),
        "missed_headings": len(expected_set - actual_set),
    }

def evaluate_accuracy():
    """
    Runs the PDF processing and evaluates the accuracy of the output against ground truth files.
    """
    project_root = Path(__file__).parent
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    ground_truth_dir = project_root / "ground_truth"

    if not ground_truth_dir.exists():
        print(f"ERROR: Ground truth directory not found at '{ground_truth_dir}'")
        print("Please create it and add manually verified JSON files.")
        return

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return

    # Step 1: Generate fresh outputs for all PDFs
    print("--- Generating fresh outputs from PDF files ---")
    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.json"
        print(f"Processing {pdf_path.name}...")
        try:
            subprocess.run(
                [sys.executable, 'generate_outputs.py', '--input', str(pdf_path), '--output', str(output_path)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  ERROR processing {pdf_path.name}:")
            print(e.stderr)
            continue
    print("---\n")

    # Step 2: Evaluate against ground truth
    print("--- Evaluating Accuracy Against Ground Truth ---")
    total_stats = {
        "files_evaluated": 0,
        "title_matches": 0,
        "total_precision": [],
        "total_recall": [],
        "total_f1": [],
    }

    for gt_path in ground_truth_dir.glob("*.json"):
        output_path = output_dir / gt_path.name
        
        if not output_path.exists():
            print(f"\n[!] WARNING: Output for {gt_path.name} not found. Skipping.")
            continue

        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        with open(output_path, 'r', encoding='utf-8') as f:
            # The script returns a JSON string inside a JSON response, so parse twice
            actual_output_str = json.load(f)
            actual_output = json.loads(actual_output_str)


        print(f"\n--- Evaluating: {gt_path.name} ---")
        total_stats["files_evaluated"] += 1

        # Compare titles
        title_match = ground_truth['title'].strip() == actual_output['title'].strip()
        if title_match:
            total_stats["title_matches"] += 1
        print(f"Title Accuracy: {'MATCH' if title_match else 'MISMATCH'}")
        if not title_match:
            print(f"  Expected: '{ground_truth['title']}'")
            print(f"  Actual:   '{actual_output['title']}'")

        # Compare headings
        heading_stats = compare_headings(actual_output['outline'], ground_truth['outline'])
        print(f"Heading Precision: {heading_stats['precision']:.2%}")
        print(f"Heading Recall:    {heading_stats['recall']:.2%}")
        print(f"Heading F1-Score:  {heading_stats['f1_score']:.2%}")
        print(f"  - Matched: {heading_stats['matched_headings']}, Missed: {heading_stats['missed_headings']}, Extra: {heading_stats['extra_headings']}")

        total_stats["total_precision"].append(heading_stats['precision'])
        total_stats["total_recall"].append(heading_stats['recall'])
        total_stats["total_f1"].append(heading_stats['f1_score'])

    # Step 3: Print overall summary
    print("\n\n--- Overall Summary ---")
    if total_stats["files_evaluated"] > 0:
        avg_title_acc = total_stats["title_matches"] / total_stats["files_evaluated"]
        avg_precision = sum(total_stats["total_precision"]) / len(total_stats["total_precision"])
        avg_recall = sum(total_stats["total_recall"]) / len(total_stats["total_recall"])
        avg_f1 = sum(total_stats["total_f1"]) / len(total_stats["total_f1"])

        print(f"Files Evaluated:   {total_stats['files_evaluated']}")
        print(f"Average Title Acc: {avg_title_acc:.2%}")
        print(f"Average Precision: {avg_precision:.2%}")
        print(f"Average Recall:    {avg_recall:.2%}")
        print(f"Average F1-Score:  {avg_f1:.2%}")
    else:
        print("No files were evaluated. Ensure ground_truth directory is populated.")

if __name__ == "__main__":
    evaluate_accuracy()
