# claude.md — AI 批改系统 · 上下文指南

> 本文件为 Claude（及其他对话式 AI 助手）提供项目快速上下文，
> 使其在无需大量探索文件系统的情况下给出准确建议。

---

## 你在帮助开发什么？

一个 **Python 数学试卷自动批改系统**，核心工作流：

1. 将 PDF 试卷转换为高分辨率 PNG
2. 用视觉语言模型（VLM）提取题目 + 学生手写答案 → 结构化 JSON
3. 用 SymPy 符号计算 + LLM 推理 进行多层正确性验证
4. 生成 Markdown 批改报告

---

## 关键文件速查

| 文件                                  | 职责                                                    |
| ------------------------------------- | ------------------------------------------------------- |
| `validation/scripts/run_pipeline.py`  | 一键运行全 Pipeline；`--step all/pdf/ocr/verify/report` |
| `validation/scripts/model_config.py`  | 所有模型 preset 配置；切换模型改此文件或设环境变量      |
| `validation/scripts/ocr_extract.py`   | VLM OCR 提取；图片自动缩放对齐 28 倍数；支持 Mock 模式  |
| `validation/scripts/math_verify.py`   | SymPy + LLM 三层验证；处理带分数/图表题/步骤验算        |
| `validation/scripts/pdf_to_images.py` | PDF → PNG 300 DPI 转换                                  |
| `math/`                               | 原始数学 PDF 试卷（9 份）                               |
| `ai_grading_analysis.md`              | 架构设计文档、模型选型、实施路线图                      |

---

## 当前已实现功能

- [x] PDF → 高分辨率 PNG（pdf2image / PyMuPDF）
- [x] VLM 结构化 OCR（支持 Ollama 本地 / DashScope / OpenAI）
- [x] Mock 模式（无需模型即可测试完整 Pipeline）
- [x] 图片自动对齐 28 倍数 + GGML 500 错误自动重试
- [x] SymPy 符号精确验证（计算题）
- [x] SymPy 步骤逐行验算（多步计算题）
- [x] 带分数多种写法规范化（又/Unicode上标/⁄）
- [x] LLM 推理批改（无标准答案时）
- [x] 图表题附带原图视觉推理
- [x] 4 阶段 JSON 容错解析（处理 Markdown / LaTeX 包裹）
- [x] Markdown 批改报告生成

---

## 模型 Preset（`model_config.py`）

```python
# 默认：本地 Ollama Qwen2.5-VL
$env:AI_MODEL_PRESET = "ollama-qwen-vl"    # Qwen2.5-VL 7B（OCR 首选，视觉）
$env:AI_MODEL_PRESET = "ollama-deepseek-ocr"
$env:AI_MODEL_PRESET = "ollama-gpt-oss"    # GPT-OSS 120B
$env:AI_MODEL_PRESET = "ollama-deepseek-v3"  # 纯文本批改
$env:AI_MODEL_PRESET = "dashscope"         # 需 DASHSCOPE_API_KEY
$env:AI_MODEL_PRESET = "openai"            # 需 OPENAI_API_KEY（GPT-4o）
```

模型配置的**唯一入口**是 `model_config.get_config(preset)`。
业务代码里不应出现任何硬编码的模型名 / base_url。

---

## 数据结构

### OCR JSON（`output/ocr_results/*_ocr.json`）

```json
{
  "source_file": "微信图片_xxx",
  "model": "qwen2.5vl:7b",
  "pages": [
    {
      "page": 1,
      "questions": [
        {
          "question_id": "第1题",
          "question_text": "计算：3x + 2x = ?",
          "student_answer": "5x",
          "answer_area_type": "计算", // 计算/填空/选择/判断
          "question_bbox": [10, 30, 60, 38], // [左%,上%,右%,下%]
          "answer_bbox": [65, 30, 95, 38]
        }
      ],
      "page_notes": ""
    }
  ]
}
```

### 验证 JSON（`output/verify_results/*_verify.json`）

```json
{
  "results": [
    {
      "question_id": "第1题",
      "is_correct": true, // true/false/null（null=需人工复核）
      "error_type": null, // "计算错误"/"答案错误"/"未作答"/null
      "answers": [
        {
          "source": "sympy", // sympy / sympy:steps / model:<name> / human
          "value": "5*x",
          "is_student_correct": true,
          "confidence": "high" // high/medium/low
        }
      ]
    }
  ]
}
```

---

## 特殊处理规则（修改时必须遵守）

### 1. 图片尺寸对齐

Qwen2.5-VL（GGML 后端）要求宽高均为 **28 的倍数**。
`ocr_extract._align_to(val, 28)` 负责此对齐，修改图片处理逻辑时必须保留。

### 2. 带分数规范化

OCR Prompt 要求模型用「**又**」格式输出带分数：`18又3/4`。
`math_verify._normalize_mixed_fraction()` 将其解析为 SymPy 可处理的 `(18+3/4)`。
扩展此函数时需同步更新 OCR Prompt 中的示例。

### 3. 多阶段 JSON 解析

模型经常在 JSON 外面包 Markdown/LaTeX，必须使用容错解析：

- `ocr_extract._parse_ocr_json(raw)`
- `math_verify._robust_json_parse(raw)`

**不要**直接用 `json.loads(response)` 替代这两个函数。

### 4. SymPy vs 模型 交叉验证

当 SymPy 步骤验算结论为「正确」但模型判断为「错误」时，
系统会将模型置信度降为 `low` 并附加警告，以 SymPy 为准。
修改 `verify_question()` 时需保留此逻辑。

### 5. 图表题视觉附图

`question_text` 含 `图/统计/柱状/折线/饼/圆/坐标` 等关键词时，
批改调用会自动附带所在页面的原始 PNG。
如需新增关键词，修改 `math_verify._VISUAL_KEYWORDS`。

---

## 常见任务示例

### 添加新的本地模型

在 `model_config.py` 的 `PRESETS` 字典新增一项，无需改其他文件：

```python
"ollama-my-model": {
    "base_url": "http://localhost:11434/v1",
    "model": "my-model:latest",
    "api_key": "ollama",
    "supports_vision": True,
    "description": "我的自定义模型",
},
```

### 调试 OCR 结果

```powershell
# Mock 模式（不调用模型，验证 Pipeline 是否通畅）
python scripts/run_pipeline.py --step ocr --mock

# 查看某个 OCR JSON
cat validation/output/ocr_results/微信图片_xxx_ocr.json
```

### 单独验证一个 JSON

```python
from pathlib import Path
from validation.scripts.math_verify import verify_ocr_results

result = verify_ocr_results(
    Path("validation/output/ocr_results/xxx_ocr.json"),
    Path("validation/output/verify_results"),
    use_model=False   # 纯 SymPy，不调模型
)
```

---

## 依赖（`validation/requirements.txt`）

```
PyMuPDF==1.27.1     # PDF 处理
Pillow==10.4.0      # 图片处理
openai>=1.0.0       # OpenAI 兼容客户端（Ollama 也用这个）
sympy==1.13.0       # 符号计算验证
tqdm==4.66.0        # 进度条
PyPDF2==3.0.1       # 备用 PDF 解析
```

---

## 开发约定

- **Python ≥ 3.10**，使用 `str | None` 而非 `Optional[str]`
- **路径**：全部用 `pathlib.Path`
- **JSON 输出**：`json.dump(..., ensure_ascii=False, indent=2)`
- **环境变量**读取在 `model_config.py` 集中管理，其他文件不应 `os.getenv` 模型相关变量
- **模型切换**：只改环境变量，不改代码

---

_Last updated: 2026-02-23_
