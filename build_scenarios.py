import os
import numpy as np
import soundfile as sf
import pandas as pd

TARGET_SR = 16000
SCENARIO_DIR = "data/scenarios"
os.makedirs(SCENARIO_DIR, exist_ok=True)

def load_clean_sample(sample_idx=0):
    manifest_path = "./data/train_set/manifest.csv"
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        clean_path = df.iloc[sample_idx % len(df)]["clean_path"]
        clean, sr = sf.read(clean_path, dtype="float32")
        if clean.ndim > 1:
            clean = np.mean(clean, axis=1)
        if len(clean) < TARGET_SR * 4:
            clean = np.pad(clean, (0, TARGET_SR * 4 - len(clean)))
        return clean[:TARGET_SR * 4]
    else:
        # Fallback synthetic speech formant track if manifest is missing
        t = np.linspace(0, 4.0, TARGET_SR * 4, endpoint=False)
        voice = (np.sin(2 * np.pi * 140 * t) * 0.4 +
                 np.sin(2 * np.pi * 280 * t) * 0.3 +
                 np.sin(2 * np.pi * 1200 * t) * 0.15)
        env = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(np.float32)
        return (voice * env).astype(np.float32)

def mix_at_snr(clean, noise, snr_db):
    c_rms = np.sqrt(np.mean(clean ** 2) + 1e-9)
    n_rms = np.sqrt(np.mean(noise ** 2) + 1e-9)
    scaled_noise = noise * (c_rms / n_rms) * (10.0 ** (-snr_db / 20.0))
    mix = clean + scaled_noise
    return np.clip(mix / (np.max(np.abs(mix)) + 1e-4) * 0.90, -0.98, 0.98)

def generate_tactical_profiles():
    print("[*] Generating 3 distinct tactical combat audio scenarios...")
    duration = 4.0
    n_samples = int(duration * TARGET_SR)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # -------------------------------------------------------------
    # 1. T-90 TANK: Deep continuous low-frequency rumble & track friction
    # -------------------------------------------------------------
    clean_tank = load_clean_sample(sample_idx=0)
    # 45 Hz engine fundamental + 90 Hz, 135 Hz, 180 Hz harmonics + broadband low-pass rumble
    tank_engine = (0.50 * np.sin(2 * np.pi * 45 * t) +
                   0.35 * np.sin(2 * np.pi * 90 * t) +
                   0.20 * np.sin(2 * np.pi * 135 * t) +
                   0.15 * np.sin(2 * np.pi * 180 * t))
    tank_track = np.convolve(np.random.randn(n_samples), np.ones(80) / 80, mode="same")
    tank_noise = tank_engine + 0.4 * tank_track
    tank_noisy = mix_at_snr(clean_tank, tank_noise, snr_db=-2.0)

    sf.write(f"{SCENARIO_DIR}/tank_clean.wav", clean_tank, TARGET_SR)
    sf.write(f"{SCENARIO_DIR}/tank_noisy.wav", tank_noisy, TARGET_SR)
    print(f"[+] Saved T-90 Tank scenario (Low-frequency drone, SNR: -2.0 dB)")

    # -------------------------------------------------------------
    # 2. ALH DHRUV HELICOPTER: Periodic 21.5 Hz rotor blade slap & downwash
    # -------------------------------------------------------------
    clean_heli = load_clean_sample(sample_idx=5)
    # 21.5 Hz periodic blade passage pulses with sharp attack
    blade_pulses = np.abs(np.sin(2 * np.pi * 21.5 * t)) ** 18
    air_turbulence = np.convolve(np.random.randn(n_samples), np.ones(35) / 35, mode="same")
    heli_noise = blade_pulses * 0.75 + air_turbulence * 0.35
    heli_noisy = mix_at_snr(clean_heli, heli_noise, snr_db=+1.5)

    sf.write(f"{SCENARIO_DIR}/heli_clean.wav", clean_heli, TARGET_SR)
    sf.write(f"{SCENARIO_DIR}/heli_noisy.wav", heli_noisy, TARGET_SR)
    print(f"[+] Saved ALH Dhruv Helicopter scenario (21.5 Hz blade slap, SNR: +1.5 dB)")

    # -------------------------------------------------------------
    # 3. ARTILLERY & GUNFIRE: Sudden ballistic shockwaves & explosions
    # -------------------------------------------------------------
    clean_combat = load_clean_sample(sample_idx=12)
    gunfire_noise = np.random.randn(n_samples) * 0.08
    # Inject 3 discrete artillery shockwaves with exponential decay
    shot_times = [0.8, 1.9, 3.1]
    for st_sec in shot_times:
        start_idx = int(st_sec * TARGET_SR)
        len_decay = int(0.45 * TARGET_SR)
        decay = np.exp(-np.linspace(0, 12, len_decay))
        blast = np.random.randn(len_decay) * decay * 2.2
        end_idx = min(start_idx + len_decay, n_samples)
        gunfire_noise[start_idx:end_idx] += blast[:end_idx - start_idx]

    combat_noisy = mix_at_snr(clean_combat, gunfire_noise, snr_db=-4.0)

    sf.write(f"{SCENARIO_DIR}/artillery_clean.wav", clean_combat, TARGET_SR)
    sf.write(f"{SCENARIO_DIR}/artillery_noisy.wav", combat_noisy, TARGET_SR)
    print(f"[+] Saved Artillery/Gunfire scenario (Impulsive shockwaves, SNR: -4.0 dB)")

    print("[SUCCESS] All 3 tactical scenario files ready in data/scenarios/")

if __name__ == "__main__":
    generate_tactical_profiles()