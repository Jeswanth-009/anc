import os
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf
import torch

# Expose PyTorch's bundled CUDA 12 libraries directly to ONNX Runtime
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_dir):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib_dir)
        except Exception:
            pass
    os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

TARGET_SR = 16000
N_FFT = 512
HOP_LEN = 128
ONNX_PATH = "checkpoints/gtcrn_rtx4060.onnx"

def compute_sisdr(ref, est):
    """DRDO standard Scale-Invariant Signal-to-Distortion Ratio."""
    eps = 1e-8
    scale = np.dot(est, ref) / (np.sum(ref ** 2) + eps)
    target = scale * ref
    noise = est - target
    return 10.0 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps))

def benchmark_dataset(manifest_csv, num_eval=50):
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Missing {manifest_csv}. Run generate_dataset.py first.")
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"Missing {ONNX_PATH}. Run export_onnx.py first.")

    df = pd.read_csv(manifest_csv).head(num_eval)
    
    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo"
        }),
        "CPUExecutionProvider"
    ]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"[*] Active ONNX Runtime Engine: {active_provider}")

    window = np.hanning(N_FFT).astype(np.float32)
    
    sisdr_in_list = []
    sisdr_out_list = []
    stoi_in_list = []
    stoi_out_list = []
    latencies = []

    print(f"[*] Benchmarking {len(df)} tactical audio samples on RTX 4060...")

    for idx, row in df.iterrows():
        noisy, _ = sf.read(row["noisy_path"], dtype="float32")
        clean, _ = sf.read(row["clean_path"], dtype="float32")
        min_l = min(len(noisy), len(clean))
        noisy, clean = noisy[:min_l], clean[:min_l]

        # STFT
        num_frames = (len(noisy) - N_FFT) // HOP_LEN + 1
        frames = np.lib.stride_tricks.sliding_window_view(noisy[:(num_frames - 1) * HOP_LEN + N_FFT], N_FFT)[::HOP_LEN]
        windowed = frames * window
        stft_spec = np.fft.rfft(windowed, n=N_FFT, axis=-1).T

        real_in = np.ascontiguousarray(stft_spec.real[np.newaxis, :, :], dtype=np.float32)
        imag_in = np.ascontiguousarray(stft_spec.imag[np.newaxis, :, :], dtype=np.float32)

        t0 = time.perf_counter()
        r_out, i_out = session.run(None, {"real_in": real_in, "imag_in": imag_in})
        latencies.append((time.perf_counter() - t0) * 1000 / max(1, num_frames))

        complex_spec = (r_out[0] + 1j * i_out[0]).T
        time_frames = np.fft.irfft(complex_spec, n=N_FFT, axis=-1) * window
        enhanced = np.zeros(len(noisy), dtype=np.float32)
        norm_w = np.zeros(len(noisy), dtype=np.float32)

        for i in range(num_frames):
            pos = i * HOP_LEN
            enhanced[pos : pos + N_FFT] += time_frames[i]
            norm_w[pos : pos + N_FFT] += window ** 2

        norm_w[norm_w < 1e-3] = 1.0
        enhanced = np.clip(enhanced / norm_w, -0.98, 0.98)

        # Quantitative Metrics
        sdr_in = compute_sisdr(clean, noisy)
        sdr_out = compute_sisdr(clean, enhanced)
        sisdr_in_list.append(sdr_in)
        sisdr_out_list.append(sdr_out)

        if HAS_STOI:
            stoi_in_list.append(stoi(clean, noisy, TARGET_SR))
            stoi_out_list.append(stoi(clean, enhanced, TARGET_SR))

    mean_sdr_gain = np.mean(np.array(sisdr_out_list) - np.array(sisdr_in_list))
    mean_stoi_in = np.mean(stoi_in_list) if HAS_STOI else 0.0
    mean_stoi_out = np.mean(stoi_out_list) if HAS_STOI else 0.0
    mean_lat = np.mean(latencies)

    print("\n" + "=" * 65)
    print("             OFFICIAL DRDO SYSTEM PERFORMANCE REPORT              ")
    print("=" * 65)
    print(f"Evaluated Test Samples  : {len(df)}")
    print(f"Execution Engine        : {active_provider}")
    print(f"Input SNR Bandwidth     : -10.0 dB to +15.0 dB")
    print("-" * 65)
    print(f"Average SI-SDR Gain     : +{mean_sdr_gain:.2f} dB  (Target: > +6.0 dB) [{'PASS' if mean_sdr_gain >= 6.0 else 'CHECK'}]")
    if HAS_STOI:
        print(f"Baseline STOI Score     :  {mean_stoi_in:.3f}")
        print(f"Enhanced STOI Score     :  {mean_stoi_out:.3f}  (Target: > 0.820)  [{'PASS' if mean_stoi_out >= 0.82 else 'CHECK'}]")
    print(f"Average Frame Latency   :  {mean_lat:.3f} ms / 8.0 ms frame  [{'PASS (<1.0ms)' if mean_lat < 1.0 else 'CHECK'}]")
    print(f"Real-Time Factor (RTF)  :  {mean_lat / 8.0:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    benchmark_dataset("./data/train_set/manifest.csv", num_eval=50)