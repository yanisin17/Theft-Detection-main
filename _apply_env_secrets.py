# -*- coding: utf-8 -*-
"""
一次性小补丁：修正 .env.example 里 VLM 段落的说明。

原文件写的是智谱 GLM（open.bigmodel.cn / glm-4v-flash），
但 vlm_filter.py 实际使用的默认是阿里云百炼 compatible-mode / qwen3-vl-flash，
两边对不上；同时原文的占位 Key 容易让人误以为有默认值。

为什么要用脚本跑：本机装了透明加密（DLP），只有白名单进程（python、git）能读到明文，
编辑器等其他工具读到的是密文，所以脚本之外的改法都改不动这个文件。

用法（在本项目根目录）：
    python _apply_env_secrets.py

执行完可以删除本文件。本脚本不包含任何密钥。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, ".env.example")

OLD = """# VLM Alert Review (visual LLM second-stage filter, fail-open)
# Fill VLM_API_KEY to enable; leave empty to disable (alerts pass through without AI review)
VLM_API_KEY=
# OpenAI-compatible endpoint. Default: Zhipu GLM (https://open.bigmodel.cn/api/paas/v4)
VLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
VLM_MODEL=glm-4v-flash
VLM_TIMEOUT=8"""

NEW = """# VLM Alert Review (visual LLM second-stage filter, fail-open)
# KEY / BASE / MODEL 三项填齐才启用；VLM_API_KEY 留空则关闭复核层，告警直接放行
VLM_API_KEY=your_vlm_api_key_here
# OpenAI 兼容接口地址（示例：阿里云百炼 compatible-mode）
VLM_API_BASE=https://YOUR-WORKSPACE.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
VLM_MODEL=qwen3-vl-flash
VLM_TIMEOUT=8"""


def main():
    if not os.path.exists(PATH):
        print("[错误] 找不到 .env.example")
        return 1

    with open(PATH, encoding="utf-8", newline="") as f:
        src = f.read()

    nl = "\r\n" if "\r\n" in src else "\n"
    new_n = NEW.replace("\n", nl)
    old_n = OLD.replace("\n", nl)

    if new_n in src:
        print("[跳过] .env.example 已经是新内容了")
        return 0

    if old_n in src:
        out = src.replace(old_n, new_n, 1)
    elif OLD in src:
        out = src.replace(OLD, NEW, 1)
    else:
        print("[失败] 没找到待替换的 VLM 段落，未做任何写入。")
        print("       请手动把 .env.example 中的 VLM 段落替换为下面这段：")
        print("-" * 60)
        print(NEW)
        print("-" * 60)
        return 1

    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(out)

    print("[完成] .env.example 已更新")
    print()
    print("当前 VLM 段落：")
    print("-" * 60)
    print(NEW)
    print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
