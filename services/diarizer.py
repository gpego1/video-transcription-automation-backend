import numpy as np
import torch
import whisper


def diarize_segments(audio_path: str, segments: list) -> list:
    if len(segments) < 2:
        return [{**seg, "speaker": "Voz 1"} for seg in segments]

    audio = whisper.load_audio(audio_path)
    sr = 16000

    embeddings = []
    valid_indices = []

    for i, seg in enumerate(segments):
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        chunk = audio[start:end]

        if len(chunk) < int(sr * 0.5):
            continue

        emb = _compute_embedding(chunk, sr)
        embeddings.append(emb)
        valid_indices.append(i)

    if len(embeddings) < 2:
        return [{**seg, "speaker": "Voz 1"} for seg in segments]

    emb_array = np.array(embeddings)
    labels = _cluster_speakers(emb_array)

    label_map = {}
    for idx, label in zip(valid_indices, labels):
        label_map[idx] = label

    return [{**seg, "speaker": f"Voz {label_map.get(i, 1)}"} for i, seg in enumerate(segments)]


def _compute_embedding(chunk, sr):
    mfcc = _compute_mfcc(chunk, sr)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    delta = np.diff(mfcc, axis=1)
    d_mean = delta.mean(axis=1) if delta.shape[1] > 0 else np.zeros_like(mean)
    d_std = delta.std(axis=1) if delta.shape[1] > 0 else np.zeros_like(mean)
    pitch = _estimate_pitch(chunk, sr)
    return np.concatenate([mean, std, d_mean, d_std, pitch])


def _compute_mfcc(chunk, sr, n_mfcc=13, n_mels=40, n_fft=512, hop=256):
    tensor = torch.from_numpy(chunk).float()
    spec = torch.stft(
        tensor, n_fft=n_fft, hop_length=hop,
        window=torch.hann_window(n_fft), return_complex=True,
    )
    power = (spec.abs() ** 2).numpy()
    mel_fb = _mel_filterbank(n_fft, n_mels, sr)
    log_mel = np.log(mel_fb @ power + 1e-10)

    dct = np.zeros((n_mfcc, n_mels))
    for k in range(n_mfcc):
        for n in range(n_mels):
            dct[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    return dct @ log_mel


def _mel_filterbank(n_fft, n_mels, sr):
    fmax = sr / 2
    mel_lo = 2595 * np.log10(1 + 0 / 700)
    mel_hi = 2595 * np.log10(1 + fmax / 700)
    mels = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz = 700 * (10 ** (mels / 2595) - 1)

    n_freqs = n_fft // 2 + 1
    fft_hz = np.linspace(0, fmax, n_freqs)
    fb = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        for j in range(n_freqs):
            if lo <= fft_hz[j] <= mid:
                fb[i, j] = (fft_hz[j] - lo) / (mid - lo + 1e-10)
            elif mid < fft_hz[j] <= hi:
                fb[i, j] = (hi - fft_hz[j]) / (hi - mid + 1e-10)
    return fb


def _estimate_pitch(chunk, sr):
    frame_len = int(sr * 0.03)
    hop = int(sr * 0.02)
    min_lag = int(sr / 400)
    max_lag = min(int(sr / 60), frame_len - 1)

    pitches = []
    total = max(1, (len(chunk) - frame_len) // hop)
    step = max(1, total // 20)

    for idx in range(0, total, step):
        start = idx * hop
        frame = chunk[start:start + frame_len].astype(np.float64)
        frame -= frame.mean()
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.01:
            continue
        frame /= rms

        n = len(frame)
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        xf = np.fft.rfft(frame, fft_size)
        acf = np.fft.irfft(xf * np.conj(xf))[:n]
        acf /= acf[0] + 1e-10

        hi = min(max_lag, len(acf) - 1)
        if min_lag >= hi:
            continue
        valid = acf[min_lag:hi]
        if len(valid) == 0:
            continue
        pk = valid.argmax()
        if valid[pk] > 0.25:
            p = sr / (pk + min_lag)
            if 60 < p < 400:
                pitches.append(p)

    if not pitches:
        return np.array([150.0, 30.0])
    return np.array([np.mean(pitches), np.std(pitches)])


def _cluster_speakers(embeddings):
    n = len(embeddings)

    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    std[std == 0] = 1
    normed = (embeddings - mean) / std

    dists = []
    dist_mx = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(normed[i] - normed[j])
            dists.append(d)
            dist_mx[i][j] = d
            dist_mx[j][i] = d

    dists = np.array(dists)

    cv = dists.std() / (dists.mean() + 1e-10)
    if cv < 0.2:
        return [1] * n

    sorted_d = np.sort(dists)
    gaps = np.diff(sorted_d)
    d_range = sorted_d[-1] - sorted_d[0]
    if d_range < 1e-10:
        return [1] * n

    rel_gaps = gaps / d_range
    max_gap_i = rel_gaps.argmax()

    if rel_gaps[max_gap_i] < 0.15:
        return [1] * n

    threshold = (sorted_d[max_gap_i] + sorted_d[max_gap_i + 1]) / 2

    labels = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if dist_mx[i][j] < threshold:
                old = labels[j]
                new = labels[i]
                for k in range(n):
                    if labels[k] == old:
                        labels[k] = new

    unique = {}
    counter = 0
    result = []
    for label in labels:
        if label not in unique:
            counter += 1
            unique[label] = counter
        result.append(unique[label])
    return result
