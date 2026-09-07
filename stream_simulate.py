import time
import numpy as np
import soundfile as sf
import torch
from gtcrn import GTCRN

FRAME_LEN = 512   # 32 ms window
HOP_LEN = 128     # 8 ms streaming step
TARGET_SR = 16000

def run_streaming_simulation(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Streaming Simulation on: {device} ({torch.cuda.get_device_name(0)})")
    
    model = GTCRN().to(device)
    model.load_state_dict(torch.load("checkpoints/gtcrn_latest.pth", map_location=device, weights_only=True))
    model.eval()

    audio, sr = sf.read(audio_path, dtype="float32")
    num_frames = (len(audio) - FRAME_LEN) // HOP_LEN

    in_buffer = np.zeros(FRAME_LEN, dtype=np.float32)
    out_buffer = np.zeros(FRAME_LEN, dtype=np.float32)
    output_audio = []
    
    window = torch.hann_window(FRAME_LEN).to(device)
    frame_latencies = []

    print(f"Streaming {num_frames} frames (each frame = 8.0 ms of tactical audio)...")

    for i in range(num_frames):
        chunk = audio[i * HOP_LEN : (i + 1) * HOP_LEN]
        
        # Shift ring buffer
        in_buffer[:-HOP_LEN] = in_buffer[HOP_LEN:]
        in_buffer[-HOP_LEN:] = chunk

        t_start = time.perf_counter()

        # 1. FFT
        frame_tensor = torch.from_numpy(in_buffer).unsqueeze(0).to(device)
        spec = torch.stft(frame_tensor, n_fft=FRAME_LEN, hop_length=HOP_LEN, window=window, return_complex=True)

        # 2. GTCRN Forward Pass
        with torch.no_grad():
            r_out, i_out = model(spec.real, spec.imag)
            est_spec = torch.complex(r_out, i_out)
            synth = torch.istft(est_spec, n_fft=FRAME_LEN, hop_length=HOP_LEN, window=window, length=FRAME_LEN)

        # 3. Overlap-Add
        out_frame = synth.squeeze().cpu().numpy()
        out_buffer += out_frame
        cleaned_step = out_buffer[:HOP_LEN].copy()
        out_buffer[:-HOP_LEN] = out_buffer[HOP_LEN:]
        out_buffer[-HOP_LEN:] = 0.0

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        frame_latencies.append(elapsed_ms)
        output_audio.extend(cleaned_step)

    avg_ms = np.mean(frame_latencies)
    p99_ms = np.percentile(frame_latencies, 99)
    rtf = avg_ms / 8.0  # Real-Time Factor (processing time / audio duration)

    print("\n" + "=" * 55)
    print("        REAL-TIME STREAMING BENCHMARK REPORT       ")
    print("=" * 55)
    print(f"Chunk Duration      : 8.00 ms (128 samples @ 16kHz)")
    print(f"Mean Execution Time : {avg_ms:.3f} ms / frame")
    print(f"99th Percentile     : {p99_ms:.3f} ms")
    print(f"Real-Time Factor    : {rtf:.4f} (Target < 0.30)")
    print(f"Latency Ceiling     : {'PASS (Deterministic)' if p99_ms < 8.0 else 'FAIL'}")
    print("=" * 55)

    sf.write("results/stream_output.wav", np.array(output_audio, dtype=np.float32), TARGET_SR)
    print("Saved stream output to: results/stream_output.wav")

if __name__ == "__main__":
    run_streaming_simulation("./data/train_set/noisy/synth_00000.wav")