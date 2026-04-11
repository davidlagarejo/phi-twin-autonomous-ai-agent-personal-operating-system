"""
mlx_runner.py — Lazy singleton MLX inference for phi-4-4bit + LoRA adapter.

Thread-safe. Loads on first call.
Adapter selection priority:
  1. mlx_ft/adapters_best/  — clean single-file dir (adapters.safetensors + adapter_config.json)
  2. mlx_ft/adapters/       — highest-numbered 0*_adapters.safetensors checkpoint
"""
import threading
from pathlib import Path

_lock   = threading.Lock()
_model  = None
_tok    = None
_loaded = False

_MLX_FT_DIR    = Path(__file__).parent.parent / "mlx_ft"
_ADAPTERS_DIR  = _MLX_FT_DIR / "adapters"
_ADAPTERS_BEST = _MLX_FT_DIR / "adapters_best"


def _base_model() -> str:
    try:
        import yaml
        cfg = yaml.safe_load((_MLX_FT_DIR / "config.yaml").read_text())
        return cfg.get("model", "mlx-community/phi-4-4bit")
    except Exception:
        return "mlx-community/phi-4-4bit"


def _resolve_adapter_dir() -> str:
    """Return the adapter directory to load.

    Prefers mlx_ft/adapters_best/ when it contains both required files,
    otherwise falls back to the highest-numbered checkpoint in mlx_ft/adapters/.
    """
    # Priority 1: clean single-file best dir
    if (
        (_ADAPTERS_BEST / "adapters.safetensors").exists()
        and (_ADAPTERS_BEST / "adapter_config.json").exists()
    ):
        return str(_ADAPTERS_BEST)

    # Priority 2: highest-numbered checkpoint in adapters/
    checkpoints = sorted(_ADAPTERS_DIR.glob("0*_adapters.safetensors"))
    if checkpoints and (_ADAPTERS_DIR / "adapter_config.json").exists():
        return str(_ADAPTERS_DIR)

    # Priority 3: canonical adapters.safetensors in adapters/
    if (_ADAPTERS_DIR / "adapters.safetensors").exists():
        return str(_ADAPTERS_DIR)

    raise FileNotFoundError(
        f"No valid adapter found in {_ADAPTERS_BEST} or {_ADAPTERS_DIR}"
    )


def load_model() -> None:
    """Force-load the model (normally called lazily on first inference)."""
    global _model, _tok, _loaded
    with _lock:
        if _loaded:
            return
        from mlx_lm import load
        base       = _base_model()
        adapter    = _resolve_adapter_dir()
        print(f"[mlx_runner] Loading {base} + adapter {adapter} …", flush=True)
        _model, _tok = load(base, adapter_path=adapter)
        _loaded = True
        print("[mlx_runner] Ready.", flush=True)


def call_mlx_sync(messages: list, max_tokens: int = 512) -> str:
    """Synchronous MLX inference.  messages = [{'role':..,'content':..}, ...]
    Thread-safe; lazy-loads on first call.
    Returns the raw text output (no JSON extraction).
    """
    load_model()

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    inner_tok = _tok._tokenizer if hasattr(_tok, "_tokenizer") else _tok
    formatted = inner_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = generate(
        _model, _tok,
        prompt=formatted,
        max_tokens=max_tokens,
        verbose=False,
        sampler=make_sampler(temp=0.1),
    )
    return raw.strip()
