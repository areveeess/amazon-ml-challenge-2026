"""
Module 4 (Part 4): Automated Validation & Final Submission Packaging.
Validates matching_results.tsv and candidate_pairs.tsv against test data using
utils/validate_submission.py, and packages the complete competition zip:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

def validate_and_package(team_name: str = "team_alpha"):
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    output_dir = project_root / "output"
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"
    test_dir = project_root / "dataset" / "test"
    validator_script = project_root / "utils" / "validate_submission.py"
    
    print(f"=== Starting Submission Verification for [{team_name}] ===")
    
    # 1. Run local validation if test files exist
    if (test_dir / "test_source1.tsv").exists():
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(matching_tsv),
            "--candidate", str(candidate_tsv),
            "--test-dir", str(test_dir)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Validation FAILED!")
            print(result.stdout)
            print(result.stderr)
            return False
        else:
            print("Validation PASSED (exit code 0)!")
    else:
        print("[Notice] Test files not found in dataset/test yet. Skipping active validation check.")
        
    # 2. Package Zip
    zip_filename = project_root / f"{team_name}_submission.zip"
    code_dir = project_root / "code" / "business_entity_resolution"
    doc_template = project_root / "Documentation_template.md"
    
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add output files
        if matching_tsv.exists():
            zipf.write(matching_tsv, arcname=f"output/{matching_tsv.name}")
        if candidate_tsv.exists():
            zipf.write(candidate_tsv, arcname=f"output/{candidate_tsv.name}")
            
        # Add documentation
        if doc_template.exists():
            zipf.write(doc_template, arcname="Documentation_template.md")
            
        # Add code directory
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith((".pyc", ".pkl")) or "__pycache__" in root:
                    continue
                file_path = Path(root) / file
                rel_path = file_path.relative_to(code_dir)
                zipf.write(file_path, arcname=f"code/business_entity_resolution/{rel_path}")
                
    print(f"\nSuccessfully generated submission archive: {zip_filename.resolve()}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="amazon_ml_team", help="Your team name")
    args = parser.parse_args()
    validate_and_package(args.team_name)
