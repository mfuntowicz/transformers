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


def to_lmir(
    model: nn.Module,
    *,
    validate: str | None = None,
) -> str:
    """Export *model* to LMIR textual IR.

    Delegates to ``model.to_lmir()`` which each supported model class
    implements via the ``LmirExportable`` protocol.

    Parameters
    ----------
    model : nn.Module
        A model whose class implements ``to_lmir() -> str``
        (e.g. ``LlamaForCausalLM``).
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
        If the model does not implement ``to_lmir``.
    RuntimeError
        If *validate* is given and ``lmir-opt`` rejects the IR.
    """
    _ensure_lmir()

    if not hasattr(model, "to_lmir"):
        raise TypeError(
            f"{type(model).__name__} does not implement to_lmir(). "
            f"LMIR export requires a model class with a to_lmir() method."
        )

    ir_text = model.to_lmir()

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
