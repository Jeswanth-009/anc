import time
import sys
import numpy as np
import sounddevice as sd
import torch
from gtcrn import GTCRN

# Audio I/O Parameters
TARGET_SR = 16000
N_FFT = 512       # 32 ms window
HOP_LEN = 128     # 8 ms tactical frame step
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tactical Noise Suppression Control
# Increase MUTE_THRESHOLD (e.g., 0.015 to 0.03) to mute more aggressive noise
MUTE_THRESHOLD = 0.012  
ATTENUATION_FLOOR = 0.02  # Drops muted background noise by 98% (-34 dB)

def initialize_engine(checkpoint_path="checkpoints/gtcrn_latest.pth"):
    print("=" * 60)
    print("   DRDO TACTICAL REAL-TIME AI NOISE CANCELLATION SYSTEM    ")
    print("=" * 60)
    print(f"[*] Initializing RTX 4060 Hardware Acceleration...")
    
    model = GTCRN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # GPU Warmup: prevents initial CUDA kernel compilation lag during live stream
    dummy_r = torch.randn(1, 257, 1, device=DEVICE)
    dummy_i = torch.randn(1, 257, 1, device=DEVICE)
    for _ in range(20):
        with torch.inference_mode():
            model(dummy_r, dummy_i)
    torch.cuda.synchronize()
    
    print(f"[+] Loaded Model: checkpoints/gtcrn_latest.pth")
    print(f"[+] CUDA Kernels Primed. Target Hop: {HOP_LEN} samples ({HOP_LEN/TARGET_SR*1000:.1f} ms)")
    return model

class RealtimeANCStream:
    def __init__(self, model):
        self.model = model
        self.window = torch.hann_window(N_FFT).to(DEVICE)
        
        # Ring buffers for Overlap-Add (OLA)
        self.in_buffer = np.zeros(N_FFT, dtype=np.float32)
        self.out_buffer = np.zeros(N_FFT, dtype=np.float32)
        self.latencies = []

    def audio_callback(self, indata, outdata, frames, time_info, status):
        t0 = time.perf_counter()
        
        # 1. Ingest incoming mono chunk (128 samples / 8ms)
        audio_chunk = indata[:, 0]
        
        # Shift input buffer
        self.in_buffer[:-HOP_LEN] = self.in_buffer[HOP_LEN:]
        self.in_buffer[-HOP_LEN:] = audio_chunk

        # 2. Real-time STFT to Complex Domain
        tensor_in = torch.from_numpy(self.in_buffer).unsqueeze(0).to(DEVICE)
        
        with torch.inference_mode():
            stft = torch.stft(tensor_in, n_fft=N_FFT, hop_length=HOP_LEN, window=self.window, return_complex=True)
            r_in, i_in = stft.real, stft.imag
            
            # Neural CRM Enhancement
            r_out, i_out = self.model(r_in, i_in)
            
            # Inverse STFT back to time domain
            est_stft = torch.complex(r_out, i_out)
            synth = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LEN, window=self.window, length=N_FFT)

        cleaned_frame = synth.squeeze().cpu().numpy()

        # 3. Dynamic Tactical Noise Gate (Mutes idle combat hiss/engine hum)
        frame_rms = np.sqrt(np.mean(cleaned_frame[-HOP_LEN:] ** 2) + 1e-9)
        if frame_rms < MUTE_THRESHOLD:
            # Soft squelch / ambient mute
            cleaned_frame[-HOP_LEN:] *= ATTENUATION_FLOOR

        # 4. Overlap-Add reconstruction
        self.out_buffer += cleaned_frame
        out_chunk = self.out_buffer[:HOP_LEN].copy()
        self.out_buffer[:-HOP_LEN] = self.out_buffer[HOP_LEN:]
        self.out_buffer[-HOP_LEN:] = 0.0

        # Prevent DAC clipping
        np.clip(out_chunk, -0.98, 0.98, out=out_chunk)
        outdata[:, 0] = out_chunk

        # Measure real-time execution headroom
        compute_time_ms = (time.perf_counter() - t0) * 1000
        self.latencies.append(compute_time_ms)
        if len(self.latencies) > 50:
            self.latencies.pop(0)

def main():
    model = initialize_engine()
    anc = RealtimeANCStream(model)

    print("\n[*] Audio Devices Detected:")
    print(f"    Input:  {sd.query_devices(kind='input')['name']}")
    print(f"    Output: {sd.query_devices(kind='output')['name']}")
    print("\n[ACTIVE] Live Tactical Noise Cancellation Engaged.")
    print("Press Ctrl+C to terminate.\n")

    # Full Duplex Stream: Input (Mic) -> Process -> Output (Headphones)
    with sd.Stream(
        samplerate=TARGET_SR,
        blocksize=HOP_LEN,     # 128 samples = 8.0 ms streaming blocks
        dtype="float32",
        channels=1,
        callback=anc.audio_callback
    ):
        try:
            while True:
                time.sleep(0.5)
                if anc.latencies:
                    avg_t = np.mean(anc.latencies)
                    rtf = avg_t / 8.0  # Execution time vs. buffer length
                    status = "REALTIME (STABLE)" if avg_t < 4.0 else "WARNING (HIGH LOAD)"
                    sys.stdout.write(f"\rRTX 4060 Frame Time: {avg_t:5.2f} ms | Budget: 8.00 ms | RTF: {rtf:5.3f} | [{status}]")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n\n[HALTED] Real-time session terminated cleanly.")

if __name__ == "__main__":
    main()