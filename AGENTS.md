# AGENTS.md — AI 批改系统

> 本文件为 AI Coding Agent（Copilot / Cursor / Gemini Code Assist 等）提供项目全局上下文，
> 阅读后可在任意文件中给出符合项目惯例的建议与修改。

---

## 项目简介

**AI 批改系统**（`ai_grading`）是一个基于 VLM + SymPy + LLM 的数学试卷自动批改 Pipeline。
它将 PDF 试卷转换为图片，使用视觉语言模型（VLM）提取题目与学生作答，再通过 SymPy 符号
计算与 LLM 推理进行多层验证，最终输出结构化批改报告。

### 支持科目（规划）

- ✅ **数学**：填空 / 计算 / 选择 / 判断（当前已实现 Pipeline）
- 🔜 **中文**：阅读理解 / 作文（待接入 PaddleOCR-VL）
- 🔜 **英文**：阅读 / 作文

---

## 目录结构

```
ai_grading/
├── math/                          # 原始数学 PDF 试卷（9 份，核心数据集）
├── validation/
│   ├── requirements.txt           # Python 依赖
│   ├── scripts/
│   │   ├── run_pipeline.py        # ★ 一键入口，协调四个步骤
│   │   ├── pdf_to_images.py       # Step 1：PDF → PNG（300 DPI）
│   │   ├── ocr_extract.py         # Step 2：VLM OCR，输出结构化 JSON
│   │   ├── math_verify.py         # Step 3：SymPy + 模型 多层验证
│   │   └── model_config.py        # 模型配置层（Ollama / DashScope / OpenAI）
│   └── output/
│       ├── images/                # Step 1 产物：PNG 图片
│       ├── ocr_results/           # Step 2 产物：*_ocr.json
│       └── verify_results/        # Step 3 产物：*_verify.json
├── validation/report/
│   └── validation_report.md       # Step 4 产物：Markdown 验证报告
├── ai_grading_analysis.md         # 技术分析报告（架构 / 模型选型 / 路线图）
└── ocr.md                         # OCR 相关记录与实验笔记
```

---

## 核心 Pipeline（四步）

```
math/*.pdf
    ↓  [Step 1] pdf_to_images.py   → output/images/<pdf_name>/page_NNN.png
    ↓  [Step 2] ocr_extract.py     → output/ocr_results/<pdf_name>_ocr.json
    ↓  [Step 3] math_verify.py     → output/verify_results/<pdf_name>_verify.json
    ↓  [Step 4] run_pipeline.py    → report/validation_report.md
```

### 快速运行（PowerShell）

```powershell
# 安装依赖
cd validation
pip install -r requirements.txt

# 使用本地 Ollama（默认，无需 API Key）
$env:AI_MODEL_PRESET = "ollama-qwen-vl"

# 全流程
python scripts/run_pipeline.py --step all

# 仅跑 OCR（无需模型，用 Mock 数据调试流程）
python scripts/run_pipeline.py --step ocr --mock

# 仅 SymPy 验证（跳过模型批改）
python scripts/run_pipeline.py --step verify --no-model

# 列出所有可用模型 preset
python scripts/run_pipeline.py --list-presets
```

---

## 模型配置规范（`model_config.py`）

所有模型通过 **preset** 统一管理，业务代码不应硬编码任何模型名称或 URL。

| Preset                  | 说明                         | 视觉支持 |
| ----------------------- | ---------------------------- | -------- |
| `ollama-qwen-vl` ← 默认 | Qwen2.5-VL 7B（本地 Ollama） | ✅       |
| `ollama-deepseek-ocr`   | DeepSeek-OCR（本地 Ollama）  | ✅       |
| `ollama-gpt-oss`        | GPT-OSS 120B（本地 Ollama）  | ✅       |
| `ollama-deepseek-v3`    | DeepSeek-V3.1 671B（纯文本） | ❌       |
| `ollama-qwen3-80b`      | Qwen3-Next 80B（纯文本）     | ❌       |
| `dashscope`             | 阿里云 Qwen2.5-VL Max        | ✅       |
| `openai`                | OpenAI GPT-4o                | ✅       |

**切换模型（环境变量优先级最高）：**

```powershell
$env:AI_MODEL_PRESET = "dashscope"
$env:DASHSCOPE_API_KEY = "sk-xxxx"
# 或完全自定义
$env:AI_BASE_URL = "http://localhost:11434/v1"
$env:AI_MODEL   = "my-custom-model"
$env:AI_API_KEY = "none"
```

---

## 数据格式规范

### OCR 输出（`*_ocr.json`）

```json
{
  "source_file": "微信图片_2026-02-03_145945_755",
  "model": "qwen2.5vl:7b",
  "pages": [
    {
      "page": 1,
      "questions": [
        {
          "question_id": "第1题",
          "question_text": "计算：(2x+3)^2 = ?",
          "student_answer": "4x^2+9",
          "answer_area_type": "计算",
          "question_bbox": [10, 30, 60, 38],
          "answer_bbox": [65, 30, 95, 38]
        }
      ],
      "page_notes": ""
    }
  ]
}
```

- `answer_area_type` 枚举：`计算` / `填空` / `选择` / `判断`
- `question_bbox` / `answer_bbox`：百分比坐标 `[左%, 上%, 右%, 下%]`
- 带分数写法：`18又3/4`（不得写成 `183/4`）
- 未作答统一为字符串 `"未作答"`

### 验证输出（`*_verify.json`）

```json
{
  "source_file": "...",
  "ocr_model": "qwen2.5vl:7b",
  "grade_model_preset": "ollama-qwen-vl",
  "total_questions": 20,
  "auto_verified": 18,
  "correct": 14,
  "incorrect": 4,
  "needs_manual_review": 2,
  "results": [
    {
      "question_id": "第1题",
      "is_correct": false,
      "error_type": "计算错误",
      "answers": [
        {
          "source": "sympy",
          "value": "4*x**2 + 12*x + 9",
          "is_student_correct": false,
          "confidence": "high",
          "note": "差值 12*x ❌"
        }
      ]
    }
  ]
}
```

---

## 验证层优先级

```
有人工标准答案？
  ├─ 计算题 → SymPy 符号计算（高置信）
  └─ 其他  → 字符串精确匹配（高置信）

无标准答案？
  ├─ 计算题 → SymPy 化简 + 步骤逐行验算，然后 LLM 批改
  └─ 其他  → LLM 批改（QWen / DeepSeek）
             └─ 若含图表关键词（图/统计/柱状等），附带原图进行视觉推理
```

置信度 `high > medium > low`，最终 `is_correct` 取置信度最高来源的结论。

---

## 编码规范

- **Python 版本**：≥ 3.10（使用 `str | None`、`match` 语句等新特性）
- **编码**：所有文件 UTF-8，JSON 输出 `ensure_ascii=False`
- **路径**：统一使用 `pathlib.Path`，不用 `os.path.join`
- **图片处理**：缩放后宽高须对齐 28 的倍数（Qwen2.5-VL GGML 要求）
- **JSON 解析**：必须经过多阶段容错（直接解析 → 去 Markdown → 清 LaTeX → 正则提取）
- **API 超时**：OCR 调用 `OCR_TIMEOUT`（默认 120s），批改调用 `GRADE_TIMEOUT`（默认 60s）
- **重试策略**：GGML/500 错误自动缩小图片尺寸重试（1260px → 980px → 700px）
- **不应**在 `ocr_extract.py` / `math_verify.py` 中硬编码模型名称，统一通过 `model_config.get_config()` 获取

---

## 扩展指南

### 新增模型 Preset

在 `model_config.py` 的 `PRESETS` 字典中添加新条目，无需改动其他文件。

### 新增科目（如语文）

1. 在 `validation/scripts/` 新建 `chinese_verify.py`，复用 `model_config` 的客户端
2. 在 `run_pipeline.py` 中添加对应步骤
3. 参考 `math_verify.verify_question` 设计验证逻辑

### 添加标准答案库

在 `verify_question()` 调用时传入 `standard_answer` 参数，系统会自动优先使用
SymPy 精确计算替代模型推理。

---

## 常见问题

| 问题                  | 原因                          | 解决                                               |
| --------------------- | ----------------------------- | -------------------------------------------------- |
| `GGML_ASSERT` 崩溃    | 图片尺寸不是 28 的倍数        | 系统已自动重试缩小；或降低 `OCR_MAX_WIDTH`         |
| JSON 解析失败         | 模型输出含 Markdown/LaTeX     | 已有 4 阶段容错，检查 `page_notes` 字段            |
| 带分数识别错误        | OCR 将 `18³⁄₄` 识别为 `183/4` | Prompt 已明确要求用「又」格式，可人工标注纠正      |
| 模型批改与 SymPy 矛盾 | 图表题缺少视觉上下文          | 确认 `images_dir` 正确传入，或提升 `GRADE_TIMEOUT` |
