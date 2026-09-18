#!/usr/bin/env python3
"""Remove specific methods from video_behavior.py by name."""

import re

path = "detection_engine/models/behavior/video_behavior.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Methods to remove completely
methods_to_remove = [
    "_extract_standard_features",
    "_extract_enhanced_features",
    "load_behavior_classifier",
    "predict_behaviors",
]

for method in methods_to_remove:
    pattern = rf"    def {method}\([^)]*\):.*?(?=\n    def |\nclass |\Z)"
    new_content = re.sub(pattern, "", content, count=1, flags=re.DOTALL)
    count = len(re.findall(pattern, content, flags=re.DOTALL))
    content = new_content
    print(f"Removed {method}: {count} occurrence(s)")

# Also remove leftover empty lines that might accumulate
content = re.sub(r"\n{3,}", "\n\n", content)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
