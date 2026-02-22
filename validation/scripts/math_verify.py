"""
math_verify.py
数学答案多层验证引擎。

验证优先级：
  1. SymPy 符号计算（计算题，精确）
  2. 模型推理批改（无标准答案时，调用 LLM 推导正确答案并判断学生作答）
  3. 字符串精确匹配（有标准答案的选择/填空）

输出结构（每道题）：
{
  "question_id": "第3题",
  "question_text": "用中文写出 20870",      <- 来自 OCR
  "answer_area_type": "填空",
  "student_answer": "二万零八百七十",
  "answers": [
    {
      "source": "sympy",                      <- sympy | model:<model_name> | human
      "value": "20870",
      "is_student_correct": true,
      "confidence": "high",                   <- high | medium | low
      "note": ""
    }
  ],
  "is_correct": true,                         <- 取优先级最高来源的结论
  "error_type": null,
  "page": 1
}
"""

import base64
import io
import json
import os
import re
import sys
from pathlib import Path

try:
    from sympy import simplify, expand, SympifyError
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
except ImportError:
    print("[ERROR] 缺少依赖: sympy")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] 缺少依赖: openai")
    sys.exit(1)

# 在同目录下找 model_config
sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_config, build_client

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# 批改用 preset：优先用纯文本大模型，也可与 OCR 共用同一模型
GRADE_PRESET = os.getenv("GRADE_MODEL_PRESET", os.getenv("AI_MODEL_PRESET", "ollama-qwen-vl"))
GRADE_TIMEOUT = int(os.getenv("GRADE_TIMEOUT", "60"))


# ─────────────────────────────────────────────
# SymPy 层
# ─────────────────────────────────────────────

def safe_parse(expr_str: str):
    if not expr_str or expr_str.strip() == "":
        return None
    cleaned = (
        expr_str.strip()
        .replace("×", "*").replace("÷", "/")
        .replace("²", "**2").replace("³", "**3")
        .replace("√", "sqrt").replace("π", "pi").replace("∞", "oo")
    )
    try:
        return parse_expr(cleaned, transformations=TRANSFORMATIONS)
    except (SympifyError, SyntaxError, TypeError, ValueError):
        return None


def sympy_check(student_ans: str, standard_ans: str | None) -> dict | None:
    """
    尝试用 SymPy 对计算题进行符号验证。
    返回 answers 条目 dict，或 None（无法完成验证时）。
    """
    s = safe_parse(student_ans)
    if s is None:
        return None

    simplified = str(expand(s))

    if standard_ans:
        c = safe_parse(standard_ans)
        if c is not None:
            try:
                diff = simplify(s - c)
                is_correct = (diff == 0)
                return {
                    "source": "sympy",
                    "value": simplified,
                    "standard_answer": standard_ans,
                    "is_student_correct": is_correct,
                    "confidence": "high",
                    "note": "符号计算等价 ✅" if is_correct else f"差值 {diff} ❌",
                }
            except Exception as e:
                pass

    # 有学生答案但无标准答案：只做化简记录
    return {
        "source": "sympy",
        "value": simplified,
        "standard_answer": None,
        "is_student_correct": None,
        "confidence": "medium",
        "note": "已化简，无标准答案可对比",
    }


def _sympy_verify_steps(student_ans: str) -> dict | None:
    """
    从学生的多步计算答案中提取每一步算式 (如 16×6×6=576)，
    逐步验证算术是否正确。返回验证报告 dict 或 None（无法提取步骤时）。
    """
    # 预处理：数字之间的 x/X 视为乘号
    text = re.sub(r'(\d)\s*[xX]\s*(\d)', r'\1×\2', student_ans)
    # 匹配形如 "16×6×6=576"  "576+576+288=1440" 的算式
    step_pattern = re.compile(
        r'([\d]+(?:[\s]*[×÷+\-*/xX][\s]*[\d]+)+)'
        r'[\s]*[=＝][\s]*'
        r'([\d]+(?:\.\d+)?)',
    )
    matches = step_pattern.findall(text)
    if not matches:
        return None

    steps = []
    all_correct = True
    for expr_str, result_str in matches:
        lhs = safe_parse(expr_str)
        rhs = safe_parse(result_str)
        if lhs is None or rhs is None:
            steps.append({"expr": f"{expr_str}={result_str}", "correct": None, "note": "无法解析"})
            continue
        try:
            diff = simplify(lhs - rhs)
            ok = (diff == 0)
        except Exception:
            ok = None
        if ok is False:
            all_correct = False
        steps.append({
            "expr": f"{expr_str}={result_str}",
            "correct": ok,
            "computed": str(lhs) if ok is False else None,
        })

    if not steps:
        return None

    n_verified = sum(1 for s in steps if s["correct"] is not None)
    n_correct = sum(1 for s in steps if s["correct"] is True)

    return {
        "source": "sympy:steps",
        "value": f"{n_correct}/{n_verified} 步正确",
        "standard_answer": None,
        "is_student_correct": all_correct if n_verified > 0 else None,
        "confidence": "high" if n_verified >= 2 else "medium",
        "note": "; ".join(
            f"{'✅' if s['correct'] else '❌'} {s['expr']}" +
            (f" (实际={s['computed']})" if s.get('computed') else "")
            for s in steps
        ),
        "steps_detail": steps,
    }


# ─────────────────────────────────────────────
# 模型批改层
# ─────────────────────────────────────────────

_grade_client: OpenAI | None = None
_grade_model: str = ""


def _get_grade_client() -> tuple[OpenAI, str]:
    global _grade_client, _grade_model
    if _grade_client is None:
        cfg = get_config(GRADE_PRESET)
        _grade_client = build_client(cfg)
        _grade_model = cfg.model
        print(f"[INFO] 批改模型: {cfg.description} ({cfg.model})")
    return _grade_client, _grade_model


def _robust_json_parse(raw: str) -> dict | None:
    """
    多阶段 JSON 解析，处理模型输出中的常见问题：
      1. 直接解析
      2. 去除 markdown 包裹后解析
      3. 清理 LaTeX 反斜杠后解析
      4. 正则提取 {...} 块后解析
    返回 dict 或 None（彻底失败时）。
    """
    if not raw:
        return None

    # 阶段 1：直接解析
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # 阶段 2：去除 markdown ```json ... ```
    cleaned = raw
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    cleaned = candidate
                    break

    # 阶段 3：清理 LaTeX 反斜杠（\(, \), \frac{}{}, \[, \] 等）
    latex_cleaned = cleaned
    # 替换常见 LaTeX 命令为纯文本
    latex_cleaned = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', latex_cleaned)
    latex_cleaned = re.sub(r'\\left|\\right', '', latex_cleaned)
    latex_cleaned = re.sub(r'\\\(|\\\)', '', latex_cleaned)  # \( \)
    latex_cleaned = re.sub(r'\\\[|\\\]', '', latex_cleaned)  # \[ \]
    latex_cleaned = re.sub(r'\\times', '×', latex_cleaned)
    latex_cleaned = re.sub(r'\\div', '÷', latex_cleaned)
    latex_cleaned = re.sub(r'\\cdot', '·', latex_cleaned)
    latex_cleaned = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', latex_cleaned)
    latex_cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', latex_cleaned)
    # 通用：清掉所有残余的 \命令名（不含花括号的）
    latex_cleaned = re.sub(r'\\([a-zA-Z]+)', r'\1', latex_cleaned)
    # 清掉单独的反斜杠
    latex_cleaned = latex_cleaned.replace('\\\\', '\n')  # \\n -> newline
    try:
        return json.loads(latex_cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 阶段 4：用正则提取最外层 {...}
    match = re.search(r'\{[\s\S]*\}', latex_cleaned)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # 阶段 4b：对原始文本也试一次正则提取
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        candidate = match.group()
        # 对提取出来的也做 LaTeX 清理
        candidate = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', candidate)
        candidate = re.sub(r'\\\(|\\\)', '', candidate)
        candidate = re.sub(r'\\\[|\\\]', '', candidate)
        candidate = re.sub(r'\\([a-zA-Z]+)', r'\1', candidate)
        candidate = candidate.replace('\\\\', '\n')
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# 视觉关键词：题目包含这些词时，应附带原图发给模型
_VISUAL_KEYWORDS = re.compile(r'图|圖|统计|統計|柱状|折线|折線|饼|圆|圓|坐标|座標|直方|示意|下表|右表|左表')


def _encode_page_image(image_path: Path) -> str | None:
    """将图片缩放到合理大小并编码为 base64 JPEG"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        max_w = 1280
        if w > max_w:
            ratio = max_w / w
            img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def model_grade(question_text: str, answer_area_type: str, student_ans: str,
                image_path: Path | None = None) -> dict:
    """
    调用 LLM 推导正确答案并判断学生作答。
    如果提供 image_path，会将该图片作为视觉上下文一起发送（用于图表题）。
    要求模型输出完整推理过程，以便人工审核模型答案的正确性。
    返回 answers 条目 dict。
    """
    has_image = image_path is not None and image_path.exists()

    image_hint = ""
    if has_image:
        image_hint = "\n\n【重要】我已附带了试卷原图，请仔细观察图片中的图表、数据、坐标等视觉信息，结合题目文字进行推理。你的推理过程必须引用从图片中读取的具体数据。"

    prompt = f"""你是一位严谨的数学老师，正在批改小学/初中数学试卷。

题目：{question_text}
题型：{answer_area_type}
学生答案：{student_ans}{image_hint}

请按以下步骤思考并输出：

1. **审题**：仔细理解题意，提取关键信息和条件{'。观察图片中的数据、图表' if has_image else ''}
2. **解题**：写出完整的解题过程（每一步都要写清楚，包括公式、计算步骤）
3. **得出正确答案**
4. **对比学生答案**：将学生答案与正确答案逐项比较
5. **判断**：给出最终判定

请严格按如下 JSON 格式输出，不要有任何其他文字：
{{
  "reasoning": "【审题】...\\n【解题过程】第1步：...\\n第2步：...\\n【正确答案】...\\n【对比】学生写的是...，正确答案是...\\n【结论】...",
  "correct_answer": "正确答案（尽量简洁）",
  "is_correct": true,
  "confidence": "high",
  "reason": "一句话总结判定理由"
}}

注意：
- reasoning 字段必须包含完整的推理链，让审核者能复现你的思路
- confidence 取值 high/medium/low，如果题目信息不足（如缺少图片）请用 medium 或 low
- 如果题目涉及图片而你没有图片信息，请在 reasoning 中说明，并将 confidence 设为 low
- 【重要】不要使用 LaTeX 公式（如 \\frac, \\(, \\)），用纯文本表示数学：1/2 而非 \\frac{{1}}{{2}}
- 求体积/面积时，学生分块计算再相加是标准方法，请先验证每块的数据是否能从图中读出、每步计算是否正确，再判断对错
- 如果学生的每一步算术都正确且最终结果合理，应判断为正确"""

    client, model = _get_grade_client()

    # 构建消息：纯文字 或 图片+文字（视觉模式）
    if has_image:
        b64 = _encode_page_image(image_path)
        if b64:
            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": prompt},
            ]}]
        else:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            timeout=GRADE_TIMEOUT,
        )
        raw = resp.choices[0].message.content.strip()
        data = _robust_json_parse(raw)
        if data is None:
            return {
                "source": f"model:{model}",
                "value": "",
                "standard_answer": None,
                "is_student_correct": None,
                "confidence": "low",
                "reasoning": "模型输出格式异常，多次解析均失败",
                "note": "JSON 解析失败，需人工复核",
                "_raw_response": raw[:500],
            }
        return {
            "source": f"model:{model}",
            "value": data.get("correct_answer", ""),
            "standard_answer": data.get("correct_answer", ""),
            "is_student_correct": data.get("is_correct"),
            "confidence": data.get("confidence", "medium"),
            "reasoning": data.get("reasoning", ""),
            "note": data.get("reason", ""),
        }
    except Exception as e:
        return {
            "source": f"model:{model}",
            "value": "",
            "standard_answer": None,
            "is_student_correct": None,
            "confidence": "low",
            "reasoning": "",
            "note": f"模型调用异常，需人工复核",
        }


# ─────────────────────────────────────────────
# 字符串匹配层（有人工标准答案时）
# ─────────────────────────────────────────────

def string_match(student_ans: str, standard_ans: str) -> dict:
    match = student_ans.strip().lower() == standard_ans.strip().lower()
    return {
        "source": "human",
        "value": standard_ans,
        "standard_answer": standard_ans,
        "is_student_correct": match,
        "confidence": "high",
        "note": "" if match else f"不匹配：学生={student_ans} 标准={standard_ans}",
    }


# ─────────────────────────────────────────────
# 主验证函数
# ─────────────────────────────────────────────

def verify_question(question: dict, standard_answer: str | None = None,
                    use_model: bool = True, image_path: Path | None = None) -> dict:
    """
    对单道题进行多层验证，返回完整的精细结果 dict。

    验证优先级：
      有人工标准答案 → string_match（选填判）/ sympy（计算）
      无标准答案 + 计算题 → sympy 化简，再 model_grade
      无标准答案 + 其他题型 → model_grade
    """
    qid = question.get("question_id", "?")
    question_text = question.get("question_text", "").strip()
    area_type = question.get("answer_area_type", "填空")
    student_ans = question.get("student_answer", "").strip()

    result = {
        "question_id": qid,
        "question_text": question_text,
        "answer_area_type": area_type,
        "student_answer": student_ans,
        "answers": [],
        "is_correct": None,
        "error_type": None,
    }

    # 未作答
    if student_ans in ("未作答", "", "？", "?"):
        result["is_correct"] = False
        result["error_type"] = "未作答"
        return result

    answers: list[dict] = []

    # 有人工标准答案
    if standard_answer:
        if area_type == "计算":
            sym = sympy_check(student_ans, standard_answer)
            if sym:
                answers.append(sym)
            else:
                # SymPy 解析失败退回字符串比较
                answers.append(string_match(student_ans, standard_answer))
        else:
            answers.append(string_match(student_ans, standard_answer))

    else:
        # 无标准答案：先 SymPy 化简 + 步骤验算（计算题），再调模型
        step_result = None
        if area_type == "计算":
            sym = sympy_check(student_ans, None)
            if sym:
                answers.append(sym)
            # 逐步验算每个算式
            step_result = _sympy_verify_steps(student_ans)
            if step_result:
                answers.append(step_result)

        if use_model and question_text:
            # 检测是否含图表关键词，若有则附带原图
            grade_image = image_path if _VISUAL_KEYWORDS.search(question_text) else None
            model_ans = model_grade(question_text, area_type, student_ans, image_path=grade_image)
            answers.append(model_ans)
            # 如果 sympy 已做了化简，把模型的标准答案填回去做对比
            if area_type == "计算" and answers and answers[0]["source"] == "sympy":
                right = model_ans.get("value", "")
                if right:
                    sym2 = sympy_check(student_ans, right)
                    if sym2 and sym2.get("is_student_correct") is not None:
                        answers[0] = sym2  # 用带标准答案的 SymPy 结果替换

            # C: 交叉验证 — SymPy 步骤全部正确但模型判错 → 以 SymPy 为准
            if (step_result
                    and step_result.get("is_student_correct") is True
                    and model_ans.get("is_student_correct") is False):
                model_ans["confidence"] = "low"
                model_ans["note"] = (
                    f"⚠️ 模型判断与 SymPy 步骤验算矛盾（{step_result['value']}），"
                    "以 SymPy 验算结果为准。" + (model_ans.get("note") or "")
                )

    # 汇总 is_correct：取置信度最高的结论
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    best = sorted(answers, key=lambda a: (confidence_rank.get(a.get("confidence", "low"), 0),), reverse=True)
    if best:
        top = best[0]
        result["is_correct"] = top.get("is_student_correct")
        if result["is_correct"] is False:
            result["error_type"] = (
                "未作答" if student_ans in ("未作答", "") else
                "计算错误" if area_type == "计算" else "答案错误"
            )

    result["answers"] = answers
    return result


# ─────────────────────────────────────────────
# OCR JSON 批量处理
# ─────────────────────────────────────────────

def verify_ocr_results(
    ocr_json_path: Path,
    output_dir: Path,
    use_model: bool = True,
    images_dir: Path | None = None,
) -> dict:
    """
    读取 OCR 结果 JSON，对每道题进行多层验证。
    如果提供 images_dir，图表题会附带原图发给批改模型进行视觉推理。
    """
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_data = json.load(f)

    source_file = ocr_data.get("source_file", ocr_json_path.stem)
    ocr_model = ocr_data.get("model", "unknown")
    verification_results = []

    # 查找每页对应的原始图片
    page_images: dict[int, Path] = {}
    if images_dir:
        source_dir = images_dir / source_file
        if source_dir.is_dir():
            for img_file in sorted(source_dir.glob("page_*.png")):
                try:
                    pnum = int(img_file.stem.split("_")[1])
                    page_images[pnum] = img_file
                except (IndexError, ValueError):
                    pass

    for page in ocr_data.get("pages", []):
        page_num = page.get("page", 0)
        page_img = page_images.get(page_num)
        for q in page.get("questions", []):
            res = verify_question(q, standard_answer=None, use_model=use_model,
                                  image_path=page_img)
            res["page"] = page_num
            verification_results.append(res)

    auto_verified = sum(1 for r in verification_results if r["is_correct"] is not None)
    output = {
        "source_file": source_file,
        "ocr_model": ocr_model,
        "grade_model_preset": GRADE_PRESET,
        "total_questions": len(verification_results),
        "auto_verified": auto_verified,
        "correct": sum(1 for r in verification_results if r["is_correct"] is True),
        "incorrect": sum(1 for r in verification_results if r["is_correct"] is False),
        "needs_manual_review": len(verification_results) - auto_verified,
        "results": verification_results,
    }

    out_path = output_dir / f"{source_file}_verify.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    correct_pct = f"{output['correct']}/{output['total_questions']}"
    print(f"  💾 {out_path.name}  ({correct_pct} 正确，{output['needs_manual_review']} 待复核)")
    return output


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数学答案多层验证")
    parser.add_argument("--no-model", action="store_true", help="跳过模型批改（纯 SymPy）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    ocr_dir = project_root / "validation" / "output" / "ocr_results"
    verify_dir = project_root / "validation" / "output" / "verify_results"
    verify_dir.mkdir(parents=True, exist_ok=True)

    if not ocr_dir.exists():
        print("[ERROR] OCR 结果目录不存在，请先运行 ocr_extract.py")
        sys.exit(1)

    for json_file in sorted(ocr_dir.glob("*_ocr.json")):
        print(f"\n[INFO] 验证: {json_file.name}")
        verify_ocr_results(json_file, verify_dir, use_model=not args.no_model)
