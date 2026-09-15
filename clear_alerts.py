import sqlite3, os, shutil

os.chdir('E:/Theft-Detection-main')

for db_name in ['theftguard.db', 'theft_detection.db']:
    if not os.path.exists(db_name):
        continue
    conn = sqlite3.connect(db_name)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print(f"\n=== {db_name} ===")
    print(f"  Tables: {tables}")
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        print(f"  {t}: {count} rows")
        if t == 'alerts' and count > 0:
            # 先删证据文件
            for row in conn.execute(f"SELECT video_path, image_path FROM [{t}]").fetchall():
                for p in row:
                    if p and os.path.exists(p):
                        os.remove(p)
                        print(f"    Deleted: {p}")
            # 删告警记录
            conn.execute(f"DELETE FROM [{t}]")
            conn.commit()
            print(f"  -> Cleared {count} alerts")
    conn.close()

# 删 evidence_videos 目录（残留帧）
if os.path.exists('evidence_videos'):
    shutil.rmtree('evidence_videos')
    print("\n  Deleted evidence_videos/ directory")
if os.path.exists('alerts'):
    shutil.rmtree('alerts')
    print("  Deleted alerts/ directory")

print("\nDone.")
