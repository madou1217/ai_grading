"""
model_config.py
统一模型配置层 — 支持 Ollama、DashScope、OpenAI 及任意 OpenAI 兼容端点。

所有模型均通过 openai 库访问，切换模型只需改一个环境变量或命令行参数，
无需修改任何业务代码。

=========================================================
快速使用（PowerShell）：
  # Ollama 本地（默认）
  $env:AI_MODEL_PRESET = "ollama-qwen-vl"
  $env:AI_MODEL_PRESET = "ollama-deepseek-ocr"
  $env:AI_MODEL_PRESET = "ollama-gpt-oss"

  # 云端 API
  $env:AI_MODEL_PRESET = "dashscope"
  $env:DASHSCOPE_API_KEY = "sk-xxx"

  $env:AI_MODEL_PRESET = "openai"
  $env:OPENAI_API_KEY = "sk-xxx"

  # 完全自定义（覆盖任意字段）
  $env:AI_BASE_URL = "http://localhost:11434/v1"
  $env:AI_MODEL   = "my-custom-model"
  $env:AI_API_KEY = "none"
=========================================================
"""

import os
from dataclasses import dataclass
from openai import OpenAI

# ─────────────────────────────────────────────────────────
# 预设配置表
# key = preset 名称（通过 AI_MODEL_PRESET 环境变量选取）
# ─────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    # ── Ollama 本地视觉模型 ───────────────────────────────
    "ollama-qwen-vl": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5vl:7b",
        "api_key": "ollama",
        "supports_vision": True,
        "description": "Ollama 本地 Qwen2.5-VL 7B（视觉，OCR 首选）",
    },
    "ollama-deepseek-ocr": {
        "base_url": "http://localhost:11434/v1",
        "model": "deepseek-ocr:latest",
        "api_key": "ollama",
        "supports_vision": True,
        "description": "Ollama 本地 DeepSeek-OCR（如有安装）",
    },
    # ── Ollama 本地大语言/视觉模型 ───────────────────────
    "ollama-gpt-oss": {
        "base_url": "http://localhost:11434/v1",
        "model": "gpt-oss:120b-cloud",
        "api_key": "ollama",
        "supports_vision": True,
        "description": "Ollama GPT-OSS 120B（大模型，综合能力强）",
    },
    "ollama-gpt-oss-20b": {
        "base_url": "http://localhost:11434/v1",
        "model": "gpt-oss:20b",
        "api_key": "ollama",
        "supports_vision": True,
        "description": "Ollama GPT-OSS 20B（轻量本地版）",
    },
    "ollama-deepseek-v3": {
        "base_url": "http://localhost:11434/v1",
        "model": "deepseek-v3.1:671b-cloud",
        "api_key": "ollama",
        "supports_vision": False,
        "description": "Ollama DeepSeek-V3.1 671B（纯文本，批改报告层）",
    },
    "ollama-qwen3-80b": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3-next:80b-cloud",
        "api_key": "ollama",
        "supports_vision": False,
        "description": "Ollama Qwen3-Next 80B（纯文本，批改推理）",
    },
    # ── 阿里云 DashScope ──────────────────────────────────
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max-latest",
        "api_key_env": "DASHSCOPE_API_KEY",
        "supports_vision": True,
        "description": "阿里云 DashScope Qwen2.5-VL Max（云端）",
    },
    # ── OpenAI 官方 ───────────────────────────────────────
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "supports_vision": True,
        "description": "OpenAI GPT-4o Vision（云端）",
    },
}

# 默认使用 Ollama Qwen2.5-VL
DEFAULT_PRESET = "ollama-qwen-vl"


@dataclass
class ModelConfig:
    """已解析并可直接使用的模型配置"""
    preset: str
    base_url: str
    model: str
    api_key: str
    supports_vision: bool
    description: str


def get_config(preset: str | None = None) -> ModelConfig:
    """
    解析并返回当前模型配置。

    优先级（从高到低）：
    1. 函数参数 preset
    2. 环境变量 AI_MODEL_PRESET
    3. 环境变量单独覆盖（AI_BASE_URL / AI_MODEL / AI_API_KEY）
    4. 默认值（ollama-qwen-vl）
    """
    preset_name = preset or os.getenv("AI_MODEL_PRESET", DEFAULT_PRESET)

    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"[model_config] 未知 preset: '{preset_name}'\n"
            f"可用 presets: {available}\n"
            f"或通过环境变量 AI_BASE_URL / AI_MODEL / AI_API_KEY 完全自定义。"
        )

    cfg = dict(PRESETS[preset_name])  # 拷贝，避免污染原始配置

    # 解析 API Key（云端 preset 从环境变量读取）
    if "api_key_env" in cfg:
        key = os.getenv(cfg["api_key_env"])
        if not key:
            raise EnvironmentError(
                f"[model_config] Preset '{preset_name}' 需要环境变量 {cfg['api_key_env']}。\n"
                f"  PowerShell: $env:{cfg['api_key_env']} = 'your-key'\n"
                f"\n  提示：使用本地 Ollama 无需 API Key，切换 preset 即可：\n"
                f"  $env:AI_MODEL_PRESET = 'ollama-qwen-vl'"
            )
        cfg["api_key"] = key

    # 允许环境变量单独覆盖任意字段（最高优先级）
    cfg["base_url"] = os.getenv("AI_BASE_URL", cfg["base_url"])
    cfg["model"]    = os.getenv("AI_MODEL", cfg["model"])
    cfg["api_key"]  = os.getenv("AI_API_KEY", cfg["api_key"])

    return ModelConfig(
        preset=preset_name,
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_key=cfg["api_key"],
        supports_vision=cfg.get("supports_vision", True),
        description=cfg.get("description", preset_name),
    )


def build_client(config: ModelConfig) -> OpenAI:
    """根据配置创建 OpenAI 兼容客户端"""
    return OpenAI(base_url=config.base_url, api_key=config.api_key)


def list_presets() -> str:
    """打印所有可用 preset"""
    lines = ["\n可用模型 Presets：", "-" * 48]
    for name, cfg in PRESETS.items():
        marker = " ← 当前默认" if name == DEFAULT_PRESET else ""
        lines.append(f"  {name:<28} {cfg['description']}{marker}")
    lines.append("")
    lines.append("切换方式：$env:AI_MODEL_PRESET = '<preset名称>'")
    lines.append("自定义  ：$env:AI_BASE_URL / AI_MODEL / AI_API_KEY")
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试：打印当前配置
    print(list_presets())
    try:
        cfg = get_config()
        print(f"\n当前配置：")
        print(f"  Preset:  {cfg.preset}")
        print(f"  描述:    {cfg.description}")
        print(f"  Base URL:{cfg.base_url}")
        print(f"  Model:   {cfg.model}")
        print(f"  视觉支持:{cfg.supports_vision}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
