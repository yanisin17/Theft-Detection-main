# -*- coding: utf-8 -*-
"""
导出报警记录为人工标注清单
==========================
用法:
    python tools/export_alerts_for_labeling.py [--db theft_detection.db] [--out labels/alerts_to_label.csv]

行为:
    1. 读取 SQLite alerts 表的全部报警记录
    2. 生成/合并 labels/alerts_to_label.csv,列:
       alert_id, timestamp, message, behavior_type, confidence, image_path,
       evidence_dir, is_true_positive, notes
    3. is_true_positive 列留空,由人工填写:
           1  = 真阳性(画面中确实发生了偷窃/藏匿行为)
           0  = 假阳性(误报)
           skip = 无法判断(截图/视频缺失等)
    4. 重复运行会保留已填写的标注,只追加新报警

填完后运行:
    python tools/eval_detection.py precision --labels labels/alerts_to_label.csv
"""

import argparse
import csv
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_FIELDS = [
    "alert_id", "timestamp", "message", "behavior_type", "confidence",
    "image_path", "evidence_dir", "is_true_positive", "notes",
]


def load_existing_labels(path):
    """读取已有人工标注(保留劳动成果),返回 {alert_id: row}"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {row["alert_id"]: row for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="导出报警记录为人工标注清单")
    parser.add_argument("--db", default=os.path.join(PROJECT_ROOT, "theft_detection.db"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "labels", "alerts_to_label.csv"))
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"错误: 找不到数据库 {args.db}")
        sys.exit(1)

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, message, timestamp, image_path, confidence, behavior_type, video_path "
        "FROM alerts ORDER BY timestamp"
    ).fetchall()
    con.close()

    existing = load_existing_labels(args.out)
    out_rows, missing_img, labeled = [], 0, 0
    for r in rows:
        alert_id = r["id"]
        image_path = r["image_path"] or ""
        if image_path and not os.path.isabs(image_path):
            image_path = os.path.join(PROJECT_ROOT, image_path)
        if image_path and not os.path.exists(image_path):
            missing_img += 1

        prev = existing.get(alert_id, {})
        label = prev.get("is_true_positive", "")
        if label:
            labeled += 1
        out_rows.append({
            "alert_id": alert_id,
            "timestamp": r["timestamp"] or "",
            "message": r["message"] or "",
            "behavior_type": r["behavior_type"] or "",
            "confidence": r["confidence"] if r["confidence"] is not None else "",
            "image_path": image_path,
            "evidence_dir": r["video_path"] or "",
            "is_true_positive": label,
            "notes": prev.get("notes", ""),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"导出 {len(out_rows)} 条报警 -> {args.out}")
    print(f"  其中已标注 {labeled} 条,待标注 {len(out_rows) - labeled} 条")
    if missing_img:
        print(f"  警告: {missing_img} 条报警的截图文件不存在(可标为 skip)")
    print("下一步: 打开 CSV 填 is_true_positive 列(1=真阳性, 0=误报, skip=无法判断),")
    print("        然后运行 python tools/eval_detection.py precision --labels " + args.out)


if __name__ == "__main__":
    main()
