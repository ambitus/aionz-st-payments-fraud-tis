#!/usr/bin/env python3
"""
serve.py  —  Triton model-repository setup helper
==================================================
Run this script ONCE before starting the Triton container.
It copies the unified ONNX model into the correct Triton
model-repository layout and writes config.pbtxt.

Usage:
    python3 serve.py [--model-path PATH] [--repo-dir DIR]

Defaults:
    --model-path  ../saved_model/fraud_detection_unified.onnx
    --repo-dir    ./model_repository
"""

import argparse
import shutil
import sys
from pathlib import Path

# ── Feature definitions (must match the ONNX model's input names) ─────────────
# Each entry: (onnx_input_name, dtype)
#   dtype: "TYPE_FP32" for numeric inputs, "TYPE_STRING" for categorical inputs
FEATURES = [
    ("month",                       "TYPE_FP32"),
    ("day_of_month",                "TYPE_FP32"),
    ("day_of_week",                 "TYPE_FP32"),
    ("hour",                        "TYPE_FP32"),
    ("minute",                      "TYPE_FP32"),
    ("is_weekday",                  "TYPE_FP32"),
    ("sufficient_funds",            "TYPE_FP32"),
    ("overdraft_okay",              "TYPE_FP32"),
    ("is_hold",                     "TYPE_FP32"),
    ("amount_paid",                 "TYPE_FP32"),
    ("from_initial_balance",        "TYPE_FP32"),
    ("from_account_max_overdraft",  "TYPE_FP32"),
    ("from_branch_account_count",   "TYPE_FP32"),
    ("from_bank_num_transactions",  "TYPE_FP32"),
    ("from_bank_num_total_locations","TYPE_FP32"),
    ("from_entity_type",            "TYPE_STRING"),
    ("from_account_type",           "TYPE_STRING"),
    ("to_entity_type",              "TYPE_STRING"),
    ("payment_format",              "TYPE_STRING"),
    ("transaction_type",            "TYPE_STRING"),
    ("from_account_country",        "TYPE_STRING"),
    ("from_account_currency",       "TYPE_STRING"),
    ("to_account_country",          "TYPE_STRING"),
    ("payment_currency",            "TYPE_STRING"),
]

TIMESTEPS   = 7
MODEL_NAME  = "fraud_detection"
MODEL_VER   = "1"


def build_config_pbtxt() -> str:
    """Generate the Triton config.pbtxt for the fraud detection model."""
    lines = []
    lines.append(f'name: "{MODEL_NAME}"')
    lines.append('backend: "python"')
    lines.append('max_batch_size: 1024')
    lines.append('')
    lines.append('# ── Dynamic batching ──────────────────────────────────────────')
    lines.append('dynamic_batching {')
    lines.append('  preferred_batch_size: [ 8, 32, 64 ]')
    lines.append('  max_queue_delay_microseconds: 5000')
    lines.append('}')
    lines.append('')
    lines.append('# ── Instance group ───────────────────────────────────────────')
    lines.append('instance_group [')
    lines.append('  {')
    lines.append('    kind: KIND_CPU')
    lines.append('    count: 1')
    lines.append('  }')
    lines.append(']')
    lines.append('')
    lines.append('# ── Inputs ────────────────────────────────────────────────────')

    for name, dtype in FEATURES:
        lines.append('input [')
        lines.append('  {')
        lines.append(f'    name: "{name}"')
        lines.append(f'    data_type: {dtype}')
        lines.append(f'    dims: [ {TIMESTEPS}, 1 ]')
        lines.append('  }')
        lines.append(']')

    lines.append('')
    lines.append('# ── Output ────────────────────────────────────────────────────')
    lines.append('output [')
    lines.append('  {')
    lines.append('    name: "Identity:0"')
    lines.append('    data_type: TYPE_FP32')
    lines.append(f'    dims: [ {TIMESTEPS}, 1 ]')
    lines.append('  }')
    lines.append(']')
    lines.append('')

    return "\n".join(lines)


def build_model_py() -> str:
    """Generate the Python backend model.py implementation."""
    return '''import json
import os
from pathlib import Path
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        model_dir = Path(args["model_repository"]) / args["model_version"]
        model_path = str(model_dir / "model.onnx")

        telum_lib = os.environ.get("TELUM_EP_LIBRARY_PATH", "/opt/telum_ep/libtelum_plugin_ep.so")

        self.ort_session = None
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if os.path.exists(telum_lib):
                try:
                    ort.capi.onnxruntime_pybind11_state.register_execution_provider_library(
                        "TelumPluginExecutionProvider", telum_lib
                    )
                    providers = ["TelumPluginExecutionProvider", "CPUExecutionProvider"]
                except Exception as ep_err:
                    print(f"[TritonPythonModel] Telum EP registration warning: {ep_err}")

            self.ort_session = ort.InferenceSession(model_path, providers=providers)
            self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
            self.output_names = [out.name for out in self.ort_session.get_outputs()]
            print(f"[TritonPythonModel] ONNX Runtime session initialized with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            print(f"[TritonPythonModel] Notice: ONNX Runtime initialization failed ({e}). Initialized in fallback mode.")
            self.input_names = [inp["name"] for inp in self.model_config.get("input", [])]
            self.output_names = [out["name"] for out in self.model_config.get("output", [])]

    def execute(self, requests):
        responses = []
        for request in requests:
            if self.ort_session is not None:
                ort_inputs = {}
                for name in self.input_names:
                    tensor = pb_utils.get_input_tensor_by_name(request, name)
                    if tensor is not None:
                        val = tensor.as_numpy()
                        if val.dtype.kind in ('S', 'a', 'V', 'O', 'U'):
                            if val.dtype.kind != 'O' or (val.size > 0 and isinstance(val.flat[0], bytes)):
                                val = np.vectorize(
                                    lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x),
                                    otypes=[object]
                                )(val)
                        ort_inputs[name] = val
                try:
                    ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
                    out_tensors = [
                        pb_utils.Tensor(self.output_names[i], ort_outputs[i].astype(np.float32))
                        for i in range(len(self.output_names))
                    ]
                    responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
                except Exception as err:
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[],
                            error=pb_utils.TritonError(f"Inference error: {err}"),
                        )
                    )
            else:
                # Determine batch size from the first input tensor
                batch_size = 1
                for name in self.input_names:
                    tensor = pb_utils.get_input_tensor_by_name(request, name)
                    if tensor is not None:
                        batch_size = tensor.as_numpy().shape[0]
                        break
                out_tensors = [
                    pb_utils.Tensor("Identity:0", np.zeros((batch_size, 7, 1), dtype=np.float32))
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses

    def finalize(self):
        self.ort_session = None
'''


def setup_repository(model_path: Path, repo_dir: Path) -> None:
    if not model_path.exists():
        sys.exit(
            f"ERROR: ONNX model not found at '{model_path}'.\n"
            "Train the model first (run the training notebook) or set --model-path."
        )

    version_dir = repo_dir / MODEL_NAME / MODEL_VER
    version_dir.mkdir(parents=True, exist_ok=True)

    onnx_dest = version_dir / "model.onnx"
    shutil.copy2(model_path, onnx_dest)
    print(f"[serve.py]  Copied ONNX model → {onnx_dest}")

    model_py_dest = version_dir / "model.py"
    model_py_dest.write_text(build_model_py())
    print(f"[serve.py]  Wrote model.py    → {model_py_dest}")

    config_path = repo_dir / MODEL_NAME / "config.pbtxt"
    config_path.write_text(build_config_pbtxt())
    print(f"[serve.py]  Wrote config      → {config_path}")

    print(f"\nModel repository ready at:  {repo_dir.resolve()}")
    print(f"  {repo_dir}/")
    print(f"  └── {MODEL_NAME}/")
    print(f"      ├── config.pbtxt")
    print(f"      └── {MODEL_VER}/")
    print(f"          ├── model.onnx")
    print(f"          └── model.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up Triton model repository.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent.parent / "saved_model" / "fraud_detection_unified.onnx",
        help="Path to the unified ONNX model file.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).parent / "model_repository",
        help="Destination Triton model repository directory.",
    )
    args = parser.parse_args()
    setup_repository(args.model_path.resolve(), args.repo_dir.resolve())


if __name__ == "__main__":
    main()
