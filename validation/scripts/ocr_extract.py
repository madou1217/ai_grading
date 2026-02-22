"""
ocr_extract.py
使用 VLM 对数学试卷图片进行结构化 OCR 识别。
模型通过 model_config.py 统一管理，支持 Ollama / DashScope / OpenAI 一键切换。

快速切换模型（PowerShell）：
  $env:AI_MODEL_PRESET = "ollama-qwen-vl"        # Qwen2.5-VL 本地
  $env:AI_MODEL_PRESET = "ollama-deepseek-ocr"   # DeepSeek-OCR 本地
  $env:AI_MODEL_PRESET = "ollama-gpt-oss"        # GPT-OSS 120B 本地
  $env:AI_MODEL_PRESET = "dashscope"             # 阿里云（需 DASHSCOPE_API_KEY）
  $env:AI_MODEL_PRESET = "openai"                # OpenAI（需 OPENAI_API_KEY）
"""

import os
import json
import base64
import io
import sys
import random
from pathlib import Path
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    print("[ERROR] 缺少依赖: Pillow")
    print("请运行: pip install Pillow")
    sys.exit(1)

try:
    from model_config import get_config, build_client, list_presets
except ImportError:
    # 当作为独立脚本运行时，确保 scripts/ 在 path 中
    sys.path.insert(0, str(Path(__file__).parent))
    from model_config import get_config, build_client, list_presets

# ─────────────────────────────────────────────
# Prompt 设计（与模型无关，统一使用）
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的数学试卷分析助手。
你的任务是从数学试卷图片中提取结构化信息。
请严格按照要求的 JSON 格式输出，不要添加任何额外解释。"""

EXTRACT_PROMPT = """分析数学试卷图片，提取题目和学生作答。

规则：
1. 识别题号、题目内容、手写答案。answer_area_type：计算/填空/选择/判断
2. 空答案填"未作答"。公式用基础符号（x^2, m/2）
3. 分数要仔细辨认分子分母，如 1/m 不要写成 1/2m。不确定的字符加[?]
4. question_bbox/answer_bbox：区域边界框[左%,上%,右%,下%]（占图片宽高的百分比0~100）

JSON格式输出：
{
  "questions": [
    {
      "question_id": "第X题",
      "question_text": "题目原文",
      "student_answer": "手写答案",
      "answer_area_type": "计算",
      "question_bbox": [10,30,60,38],
      "answer_bbox": [65,30,95,38]
    }
  ],
  "page_notes": ""
}"""

# ─────────────────────────────────────────────
# Mock 模式（无需模型，验证 Pipeline 完整流程）
# ─────────────────────────────────────────────
_MOCK_TEMPLATES = [
    {"question_id": "第1题", "question_text": "计算：(2x+3)^2 = ?", "student_answer": "4x^2+9", "answer_area_type": "计算"},
    {"question_id": "第2题", "question_text": "化简：3x + 2x - x = ?", "student_answer": "4x", "answer_area_type": "计算"},
    {"question_id": "第3题", "question_text": "求解：2x + 5 = 11，x = ?", "student_answer": "3", "answer_area_type": "填空"},
    {"question_id": "第4题", "question_text": "下列哪个是质数？A.9  B.11  C.15  D.21", "student_answer": "B", "answer_area_type": "选择"},
    {"question_id": "第5题", "question_text": "计算：sqrt(144) = ?", "student_answer": "12", "answer_area_type": "填空"},
    {"question_id": "第6题", "question_text": "两数之积为 48，之和为 14，两数各为？", "student_answer": "6和8", "answer_area_type": "计算"},
    {"question_id": "第7题", "question_text": "判断：所有偶数都是合数。（对/错）", "student_answer": "错", "answer_area_type": "判断"},
    {"question_id": "第8题", "question_text": "计算：5! = ?", "student_answer": "未作答", "answer_area_type": "计算"},
]


def mock_extract_from_image(image_path: Path) -> dict:
    """Mock 模式：生成确定性占位题目数据，不调用任何模型。"""
    random.seed(hash(str(image_path)))
    n = random.randint(4, 7)
    templates = random.sample(_MOCK_TEMPLATES, min(n, len(_MOCK_TEMPLATES)))
    questions = [dict(t, question_id=f"第{i}题") for i, t in enumerate(templates, 1)]
    return {
        "questions": questions,
        "page_notes": "[MOCK] 占位数据，非真实 OCR。",
    }


# 图片最大宽度(像素), 超过将自动缩放。
# 7B 模型建议 1024~1280px，大模型可适当调高
MAX_IMAGE_WIDTH = int(os.environ.get("OCR_MAX_WIDTH", "1280"))
# API 单次调用超时秒数
API_TIMEOUT = int(os.environ.get("OCR_TIMEOUT", "120"))


def resize_and_encode(image_path: Path, max_width: int = MAX_IMAGE_WIDTH) -> str:
    """
    读取图片，如果宽度超过 max_width 则等比例缩放，
    然后返回 JPEG base64 编码字符串。
    JPEG 常常比 PNG 小 3~5 倍，显著减少传输量。
    """
    img = Image.open(image_path)
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)
        print(f" [{w}x{h}→{img.size[0]}x{img.size[1]}]", end="", flush=True)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# 保留原 encode_image 以兼容（直接读 PNG，不缩放）
def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_from_image(client, model: str, image_path: Path) -> dict:
    """
    调用 VLM 对单张图片进行题目识别。
    图片会先缩放到 MAX_IMAGE_WIDTH 宽并转为 JPEG，显著减少皮荷和延迟。
    API 调用超时为 API_TIMEOUT 秒。
    """
    b64 = resize_and_encode(image_path)  # 自动缩放 + JPEG 压缩
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": EXTRACT_PROMPT},
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=API_TIMEOUT,  # 防止无限卡死
        )
        raw = response.choices[0].message.content.strip()

        # 多阶段 JSON 提取
        result = _parse_ocr_json(raw)
        if result is not None:
            return result

        print(f"\n[WARN] JSON 解析失败 ({image_path.name})")
        return {"questions": [], "page_notes": "JSON 解析失败，需人工复核"}

    except Exception as e:
        print(f"\n[ERROR] 模型调用失败 ({image_path.name}): {e}")
        return {"questions": [], "page_notes": f"模型错误: {str(e)[:120]}"}


def _parse_ocr_json(raw: str) -> dict | None:
    """多阶段 OCR 响应 JSON 解析"""
    import re as _re

    # 1. 直接解析
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. 去 markdown
    cleaned = raw
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            c = part.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                try:
                    return json.loads(c)
                except (json.JSONDecodeError, ValueError):
                    cleaned = c
                    break

    # 3. 正则提取 {...}
    match = _re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def run_extraction(
    images_dir: Path,
    output_dir: Path,
    mock: bool = False,
    preset: str | None = None,
) -> list[dict]:
    """
    对所有图片运行 OCR 提取，输出 JSON 结果文件。

    参数：
      images_dir  图片根目录（每个 PDF 一个子目录）
      output_dir  OCR 结果 JSON 输出目录
      mock        True = 使用占位数据（无需模型）
      preset      指定模型 preset，None 则读取 AI_MODEL_PRESET 环境变量
    """
    if mock:
        print("[INFO] MOCK 模式 — 占位数据，不调用模型\n")
        client, model = None, "mock"
    else:
        try:
            cfg = get_config(preset)
        except (ValueError, EnvironmentError) as e:
            print(f"\n{e}")
            print(list_presets())
            sys.exit(1)

        client = build_client(cfg)
        model = cfg.model
        print(f"[INFO] Preset:  {cfg.preset}")
        print(f"[INFO] 描述:    {cfg.description}")
        print(f"[INFO] Base URL:{cfg.base_url}")
        print(f"[INFO] Model:   {cfg.model}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for pdf_dir in tqdm(sorted(images_dir.iterdir()), desc="处理试卷"):
        if not pdf_dir.is_dir():
            continue

        pdf_name = pdf_dir.name
        pdf_result = {"source_file": pdf_name, "mock_mode": mock, "model": model, "pages": []}

        for img_path in sorted(pdf_dir.glob("page_*.png")):
            page_num = int(img_path.stem.split("_")[1])
            print(f"  {pdf_name}/page_{page_num:03d}...", end="", flush=True)

            page_data = mock_extract_from_image(img_path) if mock else extract_from_image(client, model, img_path)
            page_data["page"] = page_num
            pdf_result["pages"].append(page_data)
            print(f" ✅ {len(page_data.get('questions', []))} 道题")

        out_path = output_dir / f"{pdf_name}_ocr.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pdf_result, f, ensure_ascii=False, indent=2)
        all_results.append(pdf_result)
        print(f"  💾 {out_path.name}\n")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数学试卷 OCR 识别")
    parser.add_argument("--mock", action="store_true", help="使用占位数据，不调用模型")
    parser.add_argument("--preset", default=None, help="指定模型 preset（见 model_config.py）")
    parser.add_argument("--list-presets", action="store_true", help="列出所有可用 preset")
    args = parser.parse_args()

    if args.list_presets:
        print(list_presets())
        sys.exit(0)

    project_root = Path(__file__).resolve().parent.parent.parent
    images_dir = project_root / "validation" / "output" / "images"
    output_dir = project_root / "validation" / "output" / "ocr_results"

    if not images_dir.exists():
        print("[ERROR] 图片目录不存在，请先运行 pdf_to_images.py")
        sys.exit(1)

    run_extraction(images_dir, output_dir, mock=args.mock, preset=args.preset)
