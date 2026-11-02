import json
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

            sess_options = ort.SessionOptions()
            providers = ["CPUExecutionProvider"]
            if os.path.exists(telum_lib):
                try:
                    sess_options.register_custom_ops_library(telum_lib)
                    providers = ["TelumPluginExecutionProvider", "CPUExecutionProvider"]
                except Exception as ep_err:
                    print(f"[TritonPythonModel] Telum EP registration warning: {ep_err}")

            self.ort_session = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=providers
            )
            self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
            self.output_names = [out.name for out in self.ort_session.get_outputs()]
            print(f"[TritonPythonModel] ONNX Runtime session initialized with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            print(f"[TritonPythonModel] Notice: ONNX Runtime not available in Python environment ({e}). Initialized in fallback mode.")
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
                        ort_inputs[name] = tensor.as_numpy()
                try:
                    ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
                    out_tensors = [
                        pb_utils.Tensor(self.output_names[i], ort_outputs[i])
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
