"""LMIR (Language Model IR) export integration.

Provides ``to_lmir()`` for exporting transformer models to the LMIR dialect.
The generated IR uses ``!lmir.tensor`` types with named, config-bound axes
and ``#lmir.config_ref`` attributes to keep the representation parametric.

Quick start::

    from transformers import LlamaForCausalLM, LlamaConfig
    from transformers.integrations.lmir import to_lmir

    model = LlamaForCausalLM(LlamaConfig())
    ir_text = to_lmir(model)          # textual MLIR
    ir_text = to_lmir(model, validate="path/to/lmir-opt")  # also roundtrip-validates
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def _ensure_lmir():
    """Import ``lmir`` or give a helpful error."""
    try:
        import lmir  # noqa: F401
    except ImportError:
        raise ImportError(
            "The `lmir` package is required for LMIR export. "
            "Install it with:  pip install -e <path-to-lmir>/python  "
            "or add the lmir/python directory to PYTHONPATH."
        )


def _export_llama_causal_lm(model: nn.Module) -> str:
    """Export a ``LlamaForCausalLM`` to LMIR textual IR."""
    from lmir.export.builder import LmirBuilder, config_ref
    from lmir.export.shapes import ShapeContext

    config = model.config
    sc = ShapeContext(config)

    builder = LmirBuilder()
    builder.set_module_name("llama")
    builder.set_module_attr("model_type", "causal_lm")

    input_ids = builder.add_func_arg("input_ids", sc.input_ids())
    position_ids = builder.add_func_arg("position_ids", sc.position_ids())
    builder.set_func_result_types([sc.logits()])

    emb = builder.embedding(
        input_ids,
        vocab_size=config_ref("vocab_size"),
        dim=config_ref("hidden_size"),
        input_type=sc.input_ids(),
        result_type=sc.hidden(),
    )

    cos, sin = builder.rotary_embedding(
        position_ids,
        dim=config_ref("head_dim"),
        base=config_ref("rope_theta"),
        input_type=sc.position_ids(),
        cos_type=sc.rotary_cos_sin(),
        sin_type=sc.rotary_cos_sin(),
    )

    hidden = emb
    for i, _layer in enumerate(model.model.layers):
        hidden = builder.decoder_layer(
            hidden,
            cos,
            sin,
            num_heads=config_ref("num_attention_heads"),
            num_kv_heads=config_ref("num_key_value_heads"),
            head_dim=config_ref("head_dim"),
            intermediate_size=config_ref("intermediate_size"),
            rms_norm_eps=config_ref("rms_norm_eps"),
            hidden_act=config_ref("hidden_act"),
            input_type=sc.hidden(),
            cos_type=sc.rotary_cos_sin(),
            sin_type=sc.rotary_cos_sin(),
            result_type=sc.hidden(),
            name=f"layer{i}",
        )

    normed = builder.rms_norm(
        hidden,
        eps=config_ref("rms_norm_eps"),
        input_type=sc.hidden(),
        result_type=sc.hidden(),
        name="final_norm",
    )

    logits = builder.linear(
        normed,
        in_features=config_ref("hidden_size"),
        out_features=config_ref("vocab_size"),
        bias=False,
        input_type=sc.hidden(),
        result_type=sc.logits(),
        name="lm_head",
    )

    return builder.build(return_value=logits)


def _export_llama_model(model: nn.Module) -> str:
    """Export a ``LlamaModel`` to LMIR textual IR."""
    from lmir.export.builder import LmirBuilder, config_ref
    from lmir.export.shapes import ShapeContext

    config = model.config
    sc = ShapeContext(config)

    builder = LmirBuilder()
    builder.set_module_name("llama")

    input_ids = builder.add_func_arg("input_ids", sc.input_ids())
    position_ids = builder.add_func_arg("position_ids", sc.position_ids())
    builder.set_func_result_types([sc.hidden()])

    emb = builder.embedding(
        input_ids,
        vocab_size=config_ref("vocab_size"),
        dim=config_ref("hidden_size"),
        input_type=sc.input_ids(),
        result_type=sc.hidden(),
    )

    cos, sin = builder.rotary_embedding(
        position_ids,
        dim=config_ref("head_dim"),
        base=config_ref("rope_theta"),
        input_type=sc.position_ids(),
        cos_type=sc.rotary_cos_sin(),
        sin_type=sc.rotary_cos_sin(),
    )

    hidden = emb
    for i, _layer in enumerate(model.layers):
        hidden = builder.decoder_layer(
            hidden,
            cos,
            sin,
            num_heads=config_ref("num_attention_heads"),
            num_kv_heads=config_ref("num_key_value_heads"),
            head_dim=config_ref("head_dim"),
            intermediate_size=config_ref("intermediate_size"),
            rms_norm_eps=config_ref("rms_norm_eps"),
            hidden_act=config_ref("hidden_act"),
            input_type=sc.hidden(),
            cos_type=sc.rotary_cos_sin(),
            sin_type=sc.rotary_cos_sin(),
            result_type=sc.hidden(),
            name=f"layer{i}",
        )

    normed = builder.rms_norm(
        hidden,
        eps=config_ref("rms_norm_eps"),
        input_type=sc.hidden(),
        result_type=sc.hidden(),
        name="final_norm",
    )

    return builder.build(return_value=normed)


_EXPORTERS: dict[str, object] = {}


def _get_exporter(model: nn.Module):
    """Return the export function for *model*, or raise."""
    if not _EXPORTERS:
        from transformers.models.llama.modeling_llama import (
            LlamaForCausalLM,
            LlamaModel,
        )

        _EXPORTERS[LlamaForCausalLM] = _export_llama_causal_lm
        _EXPORTERS[LlamaModel] = _export_llama_model

    for cls in type(model).__mro__:
        if cls in _EXPORTERS:
            return _EXPORTERS[cls]

    supported = ", ".join(c.__name__ for c in _EXPORTERS)
    raise TypeError(
        f"No LMIR exporter for {type(model).__name__}. "
        f"Supported: {supported}"
    )


def to_lmir(
    model: nn.Module,
    *,
    validate: str | None = None,
) -> str:
    """Export *model* to LMIR textual IR.

    Parameters
    ----------
    model : nn.Module
        A supported model (currently ``LlamaForCausalLM``, ``LlamaModel``).
    validate : str, optional
        Path to the ``lmir-opt`` binary.  When provided, the generated IR is
        piped through ``lmir-opt`` to verify it parses and roundtrips.

    Returns
    -------
    str
        The MLIR module as text.

    Raises
    ------
    TypeError
        If the model type is not supported.
    RuntimeError
        If *validate* is given and ``lmir-opt`` rejects the IR.
    """
    _ensure_lmir()
    exporter = _get_exporter(model)
    ir_text = exporter(model)

    if validate:
        validate_lmir(ir_text, lmir_opt=validate)

    return ir_text


def validate_lmir(ir_text: str, *, lmir_opt: str = "lmir-opt") -> str:
    """Roundtrip *ir_text* through ``lmir-opt`` and return the output.

    Raises ``RuntimeError`` if parsing or verification fails.
    """
    result = subprocess.run(
        [lmir_opt],
        input=ir_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"lmir-opt validation failed (exit {result.returncode}):\n"
            f"{result.stderr}"
        )

    roundtrip = subprocess.run(
        [lmir_opt],
        input=result.stdout,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if roundtrip.returncode != 0:
        raise RuntimeError(
            f"lmir-opt roundtrip failed (exit {roundtrip.returncode}):\n"
            f"{roundtrip.stderr}"
        )

    if result.stdout != roundtrip.stdout:
        raise RuntimeError(
            "lmir-opt roundtrip produced different output.\n"
            f"--- first pass ---\n{result.stdout}\n"
            f"--- second pass ---\n{roundtrip.stdout}"
        )

    return result.stdout
