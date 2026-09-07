import os
import time
import torch
import numpy as np
import onnxruntime as ort
from gtcrn import GTCRN

MODEL_PATH = "checkpoints/gtcrn_latest.pth"
ONNX_OUTPUT = "checkpoints/gtcrn_rtx4060.onnx"

def export_to_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading PyTorch Model from {MODEL_PATH}...")
    
    model = GTCRN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Streaming slice shape: 1 batch, 257 frequency bins, 1 temporal frame (8.0 ms)
    dummy_real = torch.randn(1, 257, 1, device=device)
    dummy_imag = torch.randn(1, 257, 1, device=device)

    print("[*] Exporting computation graph to ONNX...")
    torch.onnx.export(
        model,
        (dummy_real, dummy_imag),
        ONNX_OUTPUT,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["real_in", "imag_in"],
        output_names=["real_out", "imag_out"],
        dynamic_axes={
            "real_in": {2: "time_frames"},
            "imag_in": {2: "time_frames"},
            "real_out": {2: "time_frames"},
            "imag_out": {2: "time_frames"}
        }
    )
    print(f"[PASS] Model successfully serialized to: {ONNX_OUTPUT}")

def benchmark_onnx_latency():
    print("\n" + "=" * 55)
    print("      ONNX RUNTIME TACTICAL LATENCY BENCHMARK      ")
    print("=" * 55)

    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE"
        }),
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(ONNX_OUTPUT, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"Active Hardware Provider: {active_provider}")

    # Single streaming frame inputs (8.0 ms chunk)
    r_dummy = np.random.randn(1, 257, 1).astype(np.float32)
    i_dummy = np.random.randn(1, 257, 1).astype(np.float32)

    # Warm-up (compile CUDA execution plans)
    for _ in range(100):
        session.run(None, {"real_in": r_dummy, "imag_in": i_dummy})

    # Benchmark 1,000 streaming frames
    runs = 1000
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {"real_in": r_dummy, "imag_in": i_dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)

    print(f"Streaming Frame Budget : 8.000 ms (128 samples @ 16kHz)")
    print(f"Mean Inference Time    : {mean_lat:.3f} ms / frame")
    print(f"95th Percentile        : {p95_lat:.3f} ms")
    print(f"99th Percentile (Worst): {p99_lat:.3f} ms")
    print(f"Real-Time Factor (RTF) : {mean_lat / 8.0:.4f}")
    print(f"Deterministic Headroom : {(8.0 - p99_lat) / 8.0 * 100:.1f}%")
    print("=" * 55)

if __name__ == "__main__":
    export_to_onnx()
    benchmark_onnx_latency()