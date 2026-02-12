import onnxruntime as ort
import numpy as np
import pickle

ONNX_PATH = "/home/xzha135/work/projects_ws/DAC/STAFI/op_097/models/supercombo.onnx"
METADATA_PATH = "/home/xzha135/work/projects_ws/DAC/STAFI/op_097/models/supercombo_metadata.pkl"


def make_session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def slice_outputs(model_outputs: np.ndarray, output_slices: dict) -> dict[str, np.ndarray]:
    return {k: model_outputs[np.newaxis, v] for k, v in output_slices.items()}


def print_outputs(parsed_outs: dict[str, np.ndarray], output_order: list) -> None:
    print("=" * 110)
    print("Parsed Model Outputs Summary")
    print("=" * 110)
    print(f"\n{'output_name':<40} {'shape':<22} {'dtype':<10} {'size':<8} {'mean':<12} {'std':<12}")
    print("-" * 105)
    for name in output_order:
        if name in parsed_outs:
            arr = parsed_outs[name]
            print(
                f"{name:<40} {str(arr.shape):<22} {str(arr.dtype):<10} {arr.size:<8} "
                f"{arr.mean():<12.6f} {arr.std():<12.6f}"
            )
    print()


def main() -> tuple[dict[str, np.ndarray], list]:
    sess = make_session(ONNX_PATH)
    out_metas = sess.get_outputs()

    with open(METADATA_PATH, "rb") as f:
        model_metadata = pickle.load(f)
    output_slices = model_metadata["output_slices"]

    inputs = {}
    for im in sess.get_inputs():
        in_name = im.name
        in_shape = im.shape
        in_type = im.type

        fixed_shape = [1 if (d is None or isinstance(d, str)) else int(d) for d in in_shape]

        if "tensor(uint8)" in in_type:
            arr = np.zeros(fixed_shape, dtype=np.uint8)
        elif "tensor(float16)" in in_type:
            arr = np.zeros(fixed_shape, dtype=np.float16)
        elif "tensor(float)" in in_type:
            arr = np.zeros(fixed_shape, dtype=np.float32)
        else:
            arr = np.zeros(fixed_shape, dtype=np.float32)
        inputs[in_name] = arr

    raw_outs = sess.run(None, inputs)
    if isinstance(raw_outs, list):
        raw_outs = {meta.name: arr for meta, arr in zip(out_metas, raw_outs)}

    raw_output = raw_outs["outputs"]
    sliced_outs = slice_outputs(raw_output[0], output_slices)
    return sliced_outs, list(output_slices.keys())


if __name__ == "__main__":
    sliced_outputs, output_order = main()
    print_outputs(sliced_outputs, output_order)
