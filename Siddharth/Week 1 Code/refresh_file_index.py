"""Refresh the ML workspace path inventory without opening data files."""
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
index = ROOT / "FILE_INDEX.csv"
summary = ROOT / "FILE_INDEX_SUMMARY.json"
errors = []
extensions = Counter()
top_levels = Counter()

# Create output paths before traversal so the inventory includes itself.
summary.touch(exist_ok=True)
with index.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["relative_path", "extension"])
    count = 0
    for directory, dirs, files in os.walk(ROOT, followlinks=False, onerror=lambda e: errors.append(str(e))):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(ROOT).as_posix()
            extension = path.suffix.lower()
            writer.writerow([relative, extension])
            extensions[extension or "(none)"] += 1
            top_levels[relative.split("/")[0] if "/" in relative else "(root)"] += 1
            count += 1

result = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "root": str(ROOT),
    "file_count": count,
    "scope": "All files enumerated recursively, including hidden files; no ignore filters; directory symlinks not followed; archives not expanded; contents not read.",
    "by_top_level": dict(sorted(top_levels.items())),
    "by_extension": dict(extensions.most_common()),
    "traversal_errors": errors,
}
summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
if errors:
    raise SystemExit(1)
