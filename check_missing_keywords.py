#!/usr/bin/env python3
"""Check .py files under 0118_oss_h200_top_20_all for missing keywords.

Shows files that don't contain:
- qk_nope_head_dim == 0
- Fast‑path for the common
- Compiled fallback (qk_nope_head_dim > 0)
"""

import re
import sys
from pathlib import Path

BASE_DIR = Path("gpu_mode/mla-decode-local/0118_oss_h200_top_20_all")

# Keywords to search for (using regex patterns to handle variations)
KEYWORDS = [
    r"qk_nope_head_dim\s*==\s*0",  # qk_nope_head_dim == 0 (with possible whitespace)
    r"Fast[‑-]path for the common",  # Fast‑path or Fast-path for the common
    r"Compiled fallback.*qk_nope_head_dim\s*>\s*0",  # Compiled fallback (qk_nope_head_dim > 0)
]

# Also check for variations
KEYWORDS_EXTENDED = [
    r"qk_nope_head_dim\s*==\s*0",
    r"Fast[‑-]path for the common",
    r"Compiled fallback.*qk_nope_head_dim\s*>\s*0",
    r"Compiled fallback.*d_nope\s*>\s*0",  # Alternative: Compiled fallback (d_nope > 0)
    r"Optimised forward for the common case.*qk_nope_head_dim\s*==\s*0",  # Alternative phrasing
]


def file_contains_keywords(file_path: Path, keywords: list) -> bool:
    """Check if file contains any of the keywords.
    
    Returns:
        True if file contains at least one keyword, False otherwise.
    """
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        for pattern in keywords:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
    return False


def main():
    import sys
    
    if not BASE_DIR.exists():
        print(f"Error: Base directory {BASE_DIR} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Find all .py files
    py_files = sorted(BASE_DIR.rglob("*.py"))
    
    if not py_files:
        print(f"No .py files found in {BASE_DIR}", file=sys.stderr)
        sys.exit(1)
    
    # Find files missing keywords
    missing_keywords = []
    
    for py_file in py_files:
        # Skip summary.tsv files if they have .py extension (unlikely but safe)
        if "summary" in py_file.name.lower():
            continue
        
        if not file_contains_keywords(py_file, KEYWORDS_EXTENDED):
            # Get relative path from BASE_DIR
            rel_path = py_file.relative_to(BASE_DIR)
            missing_keywords.append(rel_path)
    
    # Print results
    if missing_keywords:
        print(f"Found {len(missing_keywords)} files without the keywords:\n")
        for file_path in missing_keywords:
            print(file_path)
    else:
        print("All files contain at least one of the keywords.")
    
    print(f"\nTotal files checked: {len(py_files)}", file=sys.stderr)
    print(f"Files missing keywords: {len(missing_keywords)}", file=sys.stderr)


if __name__ == "__main__":
    main()
