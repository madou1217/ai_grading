"""
run_pipeline.py
AI 批改系统 - 数学试卷 OCR 验证 Pipeline 一键入口

使用方式：
  python scripts/run_pipeline.py [--step all|pdf|ocr|verify|report]

环境变量（运行前设置）：
  $env:DASHSCOPE_API_KEY = "sk-xxxx"      # 使用 Qwen2.5-VL
  # 或
  $env:OPENAI_API_KEY    = "sk-xxxx"      # 使用 GPT-4o
  $env:OCR_PROVIDER      = "openai"       # 切换提供方（默认 dashscope）
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 确保 scripts/ 目录在 sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from pdf_to_images import convert_pdfs
from ocr_extract import run_extraction
from math_verify import verify_ocr_results
from model_config import list_presets

PROJECT_ROOT = SCRIPTS_DIR.parent.parent
MATH_DIR = PROJECT_ROOT / "math"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "output"
REPORT_DIR = PROJECT_ROOT / "validation" / "report"


def step_pdf_to_images():
    print("\n" + "=" * 50)
    print("STEP 1: PDF → 图片转换")
    print("=" * 50)
    images_dir = OUTPUT_DIR / "images"
    convert_pdfs(MATH_DIR, OUTPUT_DIR, dpi=300)
    return images_dir


def step_ocr_extract(mock: bool = False, preset: str | None = None):
    print("\n" + "=" * 50)
    print("STEP 2: VLM OCR 题目识别" + (" [MOCK]" if mock else f" [{preset or 'default'}]"))
    print("=" * 50)
    images_dir = OUTPUT_DIR / "images"
    ocr_dir = OUTPUT_DIR / "ocr_results"

    if not images_dir.exists():
        print("[ERROR] 请先运行 step 1 (pdf)")
        sys.exit(1)

    return run_extraction(images_dir, ocr_dir, mock=mock, preset=preset)


def step_verify(no_model: bool = False):
    print("\n" + "=" * 50)
    suffix = " [纯SymPy]" if no_model else " [SymPy + 模型批改]"
    print(f"STEP 3: SymPy 数学验证{suffix}")
    print("=" * 50)
    ocr_dir = OUTPUT_DIR / "ocr_results"
    verify_dir = OUTPUT_DIR / "verify_results"
    images_dir = OUTPUT_DIR / "images"
    verify_dir.mkdir(parents=True, exist_ok=True)

    if not ocr_dir.exists():
        print("[ERROR] 请先运行 step 2 (ocr)")
        sys.exit(1)

    results = []
    for json_file in sorted(ocr_dir.glob("*_ocr.json")):
        print(f"\n[INFO] 验证: {json_file.name}")
        r = verify_ocr_results(json_file, verify_dir, use_model=not no_model,
                               images_dir=images_dir if images_dir.exists() else None)
        results.append(r)
    return results


def step_report(verify_results: list[dict] | None = None):
    print("\n" + "=" * 50)
    print("STEP 4: 生成验证报告")
    print("=" * 50)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取所有验证结果（如果没有传入）
    if verify_results is None:
        verify_dir = OUTPUT_DIR / "verify_results"
        verify_results = []
        for jf in sorted(verify_dir.glob("*_verify.json")):
            with open(jf, encoding="utf-8") as f:
                verify_results.append(json.load(f))

    if not verify_results:
        print("[WARN] 没有找到验证结果，请先运行 verify 步骤")
        return

    # 统计汇总
    total_q = sum(r["total_questions"] for r in verify_results)
    auto_verified = sum(r["auto_verified"] for r in verify_results)
    correct = sum(r.get("correct", 0) for r in verify_results)
    incorrect = sum(r.get("incorrect", 0) for r in verify_results)
    needs_manual = sum(r["needs_manual_review"] for r in verify_results)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ocr_model = verify_results[0].get("ocr_model", "unknown") if verify_results else "unknown"
    grade_preset = verify_results[0].get("grade_model_preset", "") if verify_results else ""

    lines = [
        f"# 数学试卷 OCR 验证报告",
        f"",
        f"> 生成时间：{now}  ",
        f"> OCR 模型：`{ocr_model}`  ",
        f"> 批改模型：`{grade_preset}`  ",
        f"> 输入试卷目录：`math/`",
        f"",
        f"---",
        f"",
        f"## 📊 总体汇总",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 处理试卷数 | {len(verify_results)} 份 |",
        f"| 识别题目总数 | {total_q} 题 |",
        f"| ✅ 正确 | {correct} 题 |",
        f"| ❌ 错误 | {incorrect} 题 |",
        f"| 🔍 待复核 | {needs_manual} 题 |",
        f"| 自动判定率 | {auto_verified/total_q*100:.1f}% |" if total_q > 0 else "| 自动判定率 | N/A |",
        f"",
        f"---",
        f"",
    ]

    for paper in verify_results:
        src = paper["source_file"]
        lines += [
            f"## 📄 {src}",
            f"",
            f"- 题目数：{paper['total_questions']}  |  "
            f"✅{paper.get('correct',0)}  ❌{paper.get('incorrect',0)}  "
            f"🔍{paper['needs_manual_review']}",
            f"",
            f"| 题号 | 类型 | 题目 | 学生答案 | 参考答案(来源) | 正确性 |",
            f"|------|------|------|---------|---------------|--------|",
        ]
        for r in paper.get("results", []):
            is_correct = (
                "✅" if r.get("is_correct") is True
                else ("❌" if r.get("is_correct") is False else "🔍")
            )
            q_text = (r.get("question_text") or "")[:40]
            s_ans = (r.get("student_answer") or "")[:25]
            # 找最高置信答案
            answers = r.get("answers", [])
            if answers:
                best = answers[0]
                ref = f"`{(best.get('value','') or '')[:20]}` ({best.get('source','')})"
            else:
                ref = "-"
            lines.append(
                f"| {r['question_id']} | {r['answer_area_type']} | "
                f"{q_text} | `{s_ans}` | {ref} | {is_correct} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 💡 下一步建议",
        "",
        "1. 对「🔍」的题目进行人工标注，补充标准答案",
        "2. 经标注后重新运行 `math_verify.py` 以获得准确率数据",
        "3. 根据错误模式分析，调整 Prompt 或微调模型",
        "",
        "*本报告由 AI 批改系统自动生成，验证结果供参考。*",
    ]

    report_path = REPORT_DIR / "validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ 报告已生成: {report_path}")
    print(f"   总计 {total_q} 道题 | 自动验证 {auto_verified} 道 | 待复核 {needs_manual} 道")


def main():
    parser = argparse.ArgumentParser(
        description="AI 批改系统 - 数学 OCR 验证 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["all", "pdf", "ocr", "verify", "report"],
        default="all",
        help="执行指定步骤（默认 all）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="OCR 使用占位数据，无需模型（验证流程用）",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="模型 preset（ollama-qwen-vl / ollama-deepseek-ocr / ollama-gpt-oss / dashscope / openai）",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="列出所有可用模型预设",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="验证步骤跳过模型批改（纯 SymPy + 字符串匹配）",
    )
    args = parser.parse_args()

    if args.list_presets:
        print(list_presets())
        sys.exit(0)

    print("\n🚀 AI 批改系统 - OCR 验证 Pipeline 启动")
    print(f"   项目根目录: {PROJECT_ROOT}")
    print(f"   数学样卷目录: {MATH_DIR}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   模式: {'MOCK' if args.mock else f'REAL [{args.preset or "AI_MODEL_PRESET env"}]'}")
    if not args.no_model:
        print(f"   批改模型: GRADE_MODEL_PRESET env 或默认")
    print()

    if args.step in ("all", "pdf"):
        step_pdf_to_images()

    if args.step in ("all", "ocr"):
        step_ocr_extract(mock=args.mock, preset=args.preset)

    verify_results = None
    if args.step in ("all", "verify"):
        verify_results = step_verify(no_model=args.no_model)

    if args.step in ("all", "report"):
        step_report(verify_results)

    print("\n🎉 Pipeline 完成！")


if __name__ == "__main__":
    main()
