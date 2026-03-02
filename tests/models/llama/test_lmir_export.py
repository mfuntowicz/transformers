"""Test LMIR export of Llama models.

Validates that LlamaForCausalLM and LlamaModel export to LMIR with
!lmir.tensor types, config_ref attributes, and optionally roundtrip
through lmir-opt.

Run with::

    LMIR_OPT=/path/to/build/bin/lmir-opt \
    PYTHONPATH=/path/to/lmir/python \
    python -m pytest tests/models/llama/test_lmir_export.py -v
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel


LMIR_OPT = os.environ.get("LMIR_OPT", shutil.which("lmir-opt"))

try:
    import lmir  # noqa: F401
    _has_lmir = True
except ImportError:
    _has_lmir = False

requires_lmir = pytest.mark.skipif(not _has_lmir, reason="lmir package not installed")
requires_lmir_opt = pytest.mark.skipif(
    LMIR_OPT is None, reason="lmir-opt not found (set LMIR_OPT env var)"
)


def _small_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


@pytest.fixture
def small_causal_lm():
    config = _small_config()
    with torch.no_grad():
        model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def small_model():
    config = _small_config()
    with torch.no_grad():
        model = LlamaModel(config)
    model.eval()
    return model


# =========================================================================
# Structure tests  -- verify the generated IR has the right shape
# =========================================================================


@requires_lmir
class TestCausalLMExport:
    def _export(self, model):
        from transformers.integrations.lmir import to_lmir
        return to_lmir(model)

    def test_module_header(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert ir.startswith("module @llama")
        assert 'lmir.model_type = "causal_lm"' in ir

    def test_func_signature(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "func.func @forward" in ir
        assert "!lmir.tensor<i64, [batch: ?, seq: ?]>" in ir

    def test_uses_lmir_tensor_types(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "!lmir.tensor<" in ir
        assert "tensor<?x?" not in ir

    def test_has_embedding(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "lmir.embedding" in ir
        assert '#lmir.config_ref<"vocab_size">' in ir

    def test_has_rotary_embedding(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "lmir.rotary_embedding" in ir
        assert '#lmir.config_ref<"head_dim">' in ir

    def test_has_decoder_layers(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        count = ir.count("lmir.decoder_layer")
        assert count == 2, f"Expected 2 decoder_layer ops, got {count}"

    def test_has_final_norm(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "lmir.rms_norm" in ir

    def test_has_lm_head(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "lmir.linear" in ir
        assert "bias = false" in ir

    def test_has_return(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "return %lm_head" in ir

    def test_no_concrete_sizes(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert "vocab_size = 256" not in ir
        assert "hidden_size = 64" not in ir

    def test_logits_result_type(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert '!lmir.tensor<f16, [batch: ?, seq: ?, vocab_size: #lmir.config_ref<"vocab_size">]>' in ir

    def test_hidden_type(self, small_causal_lm):
        ir = self._export(small_causal_lm)
        assert '!lmir.tensor<f16, [batch: ?, seq: ?, hidden_size: #lmir.config_ref<"hidden_size">]>' in ir


@requires_lmir
class TestModelExport:
    def _export(self, model):
        from transformers.integrations.lmir import to_lmir
        return to_lmir(model)

    def test_module_header(self, small_model):
        ir = self._export(small_model)
        assert ir.startswith("module @llama")

    def test_no_lm_head(self, small_model):
        ir = self._export(small_model)
        assert "lmir.linear" not in ir

    def test_returns_hidden(self, small_model):
        ir = self._export(small_model)
        assert "return %final_norm" in ir


# =========================================================================
# Validation tests  -- roundtrip through lmir-opt
# =========================================================================


@requires_lmir
@requires_lmir_opt
class TestLmirOptValidation:
    def test_causal_lm_roundtrip(self, small_causal_lm):
        from transformers.integrations.lmir import to_lmir
        ir = to_lmir(small_causal_lm, validate=LMIR_OPT)
        assert "lmir.embedding" in ir

    def test_model_roundtrip(self, small_model):
        from transformers.integrations.lmir import to_lmir
        ir = to_lmir(small_model, validate=LMIR_OPT)
        assert "lmir.rms_norm" in ir

    def test_roundtrip_preserves_tensor_types(self, small_causal_lm):
        from transformers.integrations.lmir import to_lmir, validate_lmir
        ir = to_lmir(small_causal_lm)
        roundtripped = validate_lmir(ir, lmir_opt=LMIR_OPT)
        assert "!lmir.tensor<f16" in roundtripped
        assert "!lmir.tensor<i64" in roundtripped

    def test_roundtrip_preserves_config_refs(self, small_causal_lm):
        from transformers.integrations.lmir import to_lmir, validate_lmir
        ir = to_lmir(small_causal_lm)
        roundtripped = validate_lmir(ir, lmir_opt=LMIR_OPT)
        assert '#lmir.config_ref<"hidden_size">' in roundtripped
        assert '#lmir.config_ref<"vocab_size">' in roundtripped

    def test_roundtrip_preserves_layer_count(self, small_causal_lm):
        from transformers.integrations.lmir import to_lmir, validate_lmir
        ir = to_lmir(small_causal_lm)
        roundtripped = validate_lmir(ir, lmir_opt=LMIR_OPT)
        assert roundtripped.count("lmir.decoder_layer") == 2

    def test_snapshot(self, small_causal_lm):
        """Print the full IR for manual review."""
        from transformers.integrations.lmir import to_lmir, validate_lmir
        ir = to_lmir(small_causal_lm)
        roundtripped = validate_lmir(ir, lmir_opt=LMIR_OPT)
        print("\n--- Generated LMIR (after roundtrip) ---")
        print(roundtripped)
        print("--- End LMIR ---")
