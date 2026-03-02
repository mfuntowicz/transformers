# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import Annotated

import typer


def transpile(
    model: Annotated[str, typer.Argument(help="Model ID on the Hub or local path to a model directory.")],
    target: Annotated[str, typer.Argument(help="Code generation target (e.g. 'mlx').")],
    output: Annotated[
        Path,
        typer.Argument(help="Directory to write the generated file(s) into."),
    ],
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Allow custom models defined on the Hub in their own modeling files."),
    ] = False,
    lmir_opt: Annotated[
        str | None,
        typer.Option(help="Path to `lmir-opt` binary for IR validation. Skips validation when omitted."),
    ] = None,
    emit_ir: Annotated[
        bool,
        typer.Option(help="Also write the intermediate LMIR IR alongside the generated code."),
    ] = False,
):
    """Transpile a model to a target framework via LMIR.

    Exports the model to LMIR IR, then generates source code for the
    specified target (currently: mlx).

    \b
    Examples
    --------
      transformers transpile meta-llama/Llama-3.1-8B mlx ./out/
      transformers transpile ./my-local-llama mlx ./out/ --lmir-opt build/bin/lmir-opt
    """
    import torch

    from ..integrations.lmir import to_lmir
    from ..models.auto import AutoModelForCausalLM

    try:
        from lmir.codegen import generate
    except ImportError:
        typer.echo(
            "Error: the `lmir` package is required for transpilation. "
            "Install it or add lmir/python to PYTHONPATH.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loading model {model!r}…")
    with torch.no_grad():
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
        )

    validate = lmir_opt if lmir_opt else None
    typer.echo("Exporting to LMIR IR…")
    ir_text = to_lmir(hf_model, validate=validate)

    typer.echo(f"Generating {target} code…")
    code = generate(ir_text, target=target)

    output.mkdir(parents=True, exist_ok=True)
    code_path = output / "model.py"
    code_path.write_text(code)
    typer.echo(f"Written {code_path}")

    if emit_ir:
        ir_path = output / "model.mlir"
        ir_path.write_text(ir_text)
        typer.echo(f"Written {ir_path}")

    typer.echo("Done.")
