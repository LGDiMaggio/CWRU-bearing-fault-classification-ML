import numpy as np

def extract_time_domain_features(signal):
    """Extracts time-domain features from a signal."""
    N = len(signal)
    p1 = np.mean(signal)  # Mean
    p2 = np.std(signal, ddof=1)  # Standard deviation
    p3 = np.mean(np.abs(signal - p1))  # Mean absolute deviation
    p4 = np.sqrt(np.mean(np.square(signal)))  # Root mean square
    p5 = np.max(np.abs(signal))  # Maximum absolute value
    p6 = np.sum((signal - p1)**3) / ((N - 1) * p2**3)  # Skewness
    p7 = np.sum((signal - p1)**4) / ((N - 1) * p2**4)  # Kurtosis
    p8 = p5 / p4  # Crest factor
    p9 = p5 / p3  # Form factor
    p10 = p4 / p3  # Shape factor
    p11 = p4**2 / np.mean(np.abs(signal))  # Impulse factor

    features = {
        'Mean': p1,
        'Standard deviation': p2,
        'Mean absolute deviation': p3,
        'Root mean square': p4,
        'Maximum absolute value': p5,
        'Skewness': p6,
        'Kurtosis': p7,
        'Crest factor': p8,
        'Form factor': p9,
        'Shape factor': p10,
        'Impulse factor': p11
    }
    return features

def extract_frequency_domain_features(signal, sampling_rate=48000):
    """Extracts frequency-domain features from a signal."""
    L = len(signal)
    fft_vals = np.fft.fft(signal)
    P2 = np.abs(fft_vals / L)  # Normalized FFT
    ps = P2[:L//2]
    ps[1:-1] *= 2  # Compensating for the symmetry
    K = len(ps)
    freqs = np.fft.fftfreq(L, d=1/sampling_rate)[:L//2]

    p12 = np.mean(ps)
    p13 = np.std(ps, ddof=1)
    p14 = np.sum((ps - p12)**3) / (K * p13**3)
    p15 = np.sum((ps - p12)**4) / (K * p13**4) - 3
    p16 = np.sum(ps * freqs) / np.sum(ps)
    p17 = np.sqrt(np.sum((freqs - p16)**2 * ps) / K)
    p18 = np.sqrt(np.sum(freqs**2 * ps) / np.sum(ps))
    p19 = np.sqrt(np.sum(freqs**4 * ps) / np.sum(freqs**2 * ps))
    p20 = np.sqrt((np.sum(ps * freqs) - np.sum(ps) * p17) / np.sum(ps))
    p21 = p17 / p16 if p16 != 0 else 0
    p22 = np.sum(freqs * (ps - p12)**3 * ps) / (K * p13 * p17)
    p23 = np.sum(freqs * (ps - p12)**4 * ps) / (K * p13**2 * p17)

    features = {
        'Mean of power spectrum': p12,
        'Standard deviation of power spectrum': p13,
        'Skewness of power spectrum': p14,
        'Kurtosis of power spectrum': p15,
        'Mean frequency': p16,
        'Standard deviation of frequency': p17,
        'Root mean square frequency': p18,
        'Root variance frequency': p19,
        'Frequency centroid': p20,
        'Frequency variation factor': p21,
        'Frequency variance': p22,
        'Frequency skewness': p23
    }
    return features