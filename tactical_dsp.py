import numpy as np

class AcousticBlastLimiter:
    """
    Zero-lookahead tactical transient limiter.
    Detects ballistic shockwaves and muzzle blasts using dual-time-constant
    peak tracking, applying soft-knee hyperbolic saturation to prevent
    ADC clipping and GRU hidden-state saturation.
    """
    def __init__(self, sample_rate=16000, threshold_db=-3.0, attack_ms=0.2, release_ms=30.0):
        self.sr = sample_rate
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self.attack_coeff = np.exp(-1.0 / (attack_ms * 1e-3 * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release_ms * 1e-3 * sample_rate))
        self.envelope = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            s = np.abs(x[i])
            # Zero-lookahead envelope detector
            if s > self.envelope:
                self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * s
            else:
                self.envelope = self.release_coeff * self.envelope + (1.0 - self.release_coeff) * s

            # Soft-knee compression via tanh above threshold
            if self.envelope > self.threshold:
                gain = self.threshold / (self.envelope + 1e-8)
                comp = x[i] * gain
                out[i] = np.tanh(comp / self.threshold) * self.threshold
            else:
                out[i] = x[i]

        return np.clip(out, -0.98, 0.98)


class TacticalDualMicBeamformer:
    """
    Simulates dual-transducer tactical headsets (e.g., Peltor ComTac / Bose TTH).
    - Mic 1 (Boom Mic): Near-field speech + ambient noise (mouth proximity: ~2 cm).
    - Mic 2 (Earcup Mic): Ambient diffuse noise reference + heavily attenuated speech (~14 cm distance).
    Applies adaptive differential spatial cancellation to strip 3-6 dB of diffuse noise
    prior to neural spectral masking.
    """
    def __init__(self, filter_order=32, mu=0.015, eps=1e-6):
        self.order = filter_order
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(filter_order, dtype=np.float32)
        self.ref_buffer = np.zeros(filter_order, dtype=np.float32)

    def synthesize_array_from_mono(self, clean_speech: np.ndarray, combat_noise: np.ndarray, snr_db: float):
        """Generates synthetic 2-channel acoustic field matching tactical headset geometry."""
        # Scale noise power for target SNR on primary boom mic
        clean_rms = np.sqrt(np.mean(clean_speech**2) + 1e-9)
        noise_rms = np.sqrt(np.mean(combat_noise**2) + 1e-9)
        noise_scaled = combat_noise * (clean_rms / noise_rms) * (10.0 ** (-snr_db / 20.0))

        # Channel 1 (Boom Mic): Direct vocal path (0 dB attenuation, 0 delay) + Ambient Noise
        mic_boom = clean_speech + noise_scaled

        # Channel 2 (Earcup Mic): Speech attenuated by ~18 dB (inverse square law at 14 cm) +
        # uncorrelated ambient noise with small spatial phase lag (5 samples = ~0.3 ms delay)
        speech_attenuated = clean_speech * 0.12
        noise_earcup = np.roll(noise_scaled, shift=5)

        mic_earcup = speech_attenuated + noise_earcup
        return mic_boom.astype(np.float32), mic_earcup.astype(np.float32)

    def filter(self, boom_ch: np.ndarray, earcup_ch: np.ndarray) -> np.ndarray:
        """Adaptive Generalized Sidelobe Canceller (GSC) spatial stage."""
        length = len(boom_ch)
        beamformed = np.zeros(length, dtype=np.float32)

        for n in range(length):
            self.ref_buffer[1:] = self.ref_buffer[:-1]
            self.ref_buffer[0] = earcup_ch[n]

            est_noise = np.dot(self.weights, self.ref_buffer)
            err = boom_ch[n] - est_noise
            beamformed[n] = err

            # Normalized stochastic gradient update
            pwr = np.dot(self.ref_buffer, self.ref_buffer) + self.eps
            self.weights += (self.mu / pwr) * err * self.ref_buffer

        return np.clip(beamformed, -0.98, 0.98)