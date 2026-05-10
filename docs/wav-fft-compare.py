#!/usr/bin/env python3
"""b140.7.5: compare FFT of an exported PhaseForge WAV with what the
PhaseForge UI reports on its Export tab. Helps diagnose whether the
exported impulse is actually correct (REW reader bug) or whether the
PhaseForge plot is misleading (impulse really is broken).

Usage:
    python3 docs/wav-fft-compare.py <path_to_wav>

Tries scipy.io.wavfile first; falls back to a manual WAV parser if scipy
isn't installed or the format isn't supported (e.g. 64-bit IEEE float,
which scipy can't read out of the box).
"""
import struct
import sys


def manual_parse(path):
    """Tiny WAV parser supporting PCM int16/24/32 and IEEE float 32/64."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError("not a RIFF/WAVE file")
    # walk chunks after WAVE marker
    fmt = None
    data_chunk = None
    i = 12
    while i + 8 <= len(data):
        cid = data[i:i + 4]
        clen = struct.unpack_from("<I", data, i + 4)[0]
        body = data[i + 8:i + 8 + clen]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data_chunk = body
        i += 8 + clen + (clen & 1)  # pad to even
    if fmt is None or data_chunk is None:
        raise RuntimeError("missing fmt/data chunk")

    fmt_code = struct.unpack_from("<H", fmt, 0)[0]
    channels = struct.unpack_from("<H", fmt, 2)[0]
    sr = struct.unpack_from("<I", fmt, 4)[0]
    bps = struct.unpack_from("<H", fmt, 14)[0]
    total_samples = len(data_chunk) // (bps // 8) // channels

    print(f"[WAV header] fmt_code={fmt_code} channels={channels} sr={sr} bps={bps}")
    print(f"[WAV header] data bytes={len(data_chunk)} samples_per_ch={total_samples}")

    if fmt_code == 3 and bps == 64:
        out = struct.unpack(f"<{len(data_chunk) // 8}d", data_chunk)
    elif fmt_code == 3 and bps == 32:
        out = struct.unpack(f"<{len(data_chunk) // 4}f", data_chunk)
    elif fmt_code == 1 and bps == 16:
        ints = struct.unpack(f"<{len(data_chunk) // 2}h", data_chunk)
        out = [v / 32768.0 for v in ints]
    elif fmt_code == 1 and bps == 32:
        ints = struct.unpack(f"<{len(data_chunk) // 4}i", data_chunk)
        out = [v / 2147483648.0 for v in ints]
    else:
        raise RuntimeError(f"unsupported format: code={fmt_code} bps={bps}")

    # de-interleave: take channel 0
    if channels > 1:
        out = list(out[0::channels])
    return sr, list(out)


def main():
    if len(sys.argv) != 2:
        print("Usage: wav-fft-compare.py <path>")
        sys.exit(1)
    path = sys.argv[1]

    sr = None
    data = None
    # Try scipy first (cleaner, supports common formats).
    try:
        import scipy.io.wavfile as wav  # type: ignore
        import numpy as np  # type: ignore
        sr, raw = wav.read(path)
        if raw.ndim > 1:
            raw = raw[:, 0]
        if raw.dtype.kind in ("i", "u"):
            max_val = float(np.iinfo(raw.dtype).max)
            data = (raw.astype(np.float64) / max_val).tolist()
        else:
            data = raw.astype(float).tolist()
        print(f"[scipy] dtype={raw.dtype}")
    except Exception as e:
        print(f"[scipy] fallback to manual parser: {e}")
        sr, data = manual_parse(path)

    n = len(data)
    print(f"[WAV] sr={sr}, samples={n}")
    print(f"[WAV] impulse[0..5]={data[:5]}")
    peak = max(abs(v) for v in data)
    s = sum(data)
    print(f"[WAV] peak_abs={peak:.6e}")
    print(f"[WAV] sum={s:.6e}")

    # FFT magnitude in dB
    try:
        import numpy as np  # type: ignore
    except ImportError:
        print("numpy not installed — cannot run FFT", file=sys.stderr)
        sys.exit(2)

    arr = np.asarray(data, dtype=np.float64)
    spec = np.fft.rfft(arr)
    mag = np.abs(spec)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-300))

    # Normalise the *peak* to 0 dB so absolute level doesn't confuse the
    # comparison (PhaseForge UI does the same on its plot).
    peak_db = float(np.max(mag_db))
    mag_db_norm = mag_db - peak_db
    print(f"[FFT] peak_mag_db (pre-norm) = {peak_db:+.2f}")

    probes = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
    print()
    print(f"{'Freq (Hz)':>10}  {'mag_db (raw)':>14}  {'mag_db (peak=0)':>16}")
    for f in probes:
        if f >= sr / 2:
            print(f"{f:>10}  {'> Nyquist':>14}")
            continue
        idx = round(f * n / sr)
        print(f"{f:>10}  {mag_db[idx]:>+14.2f}  {mag_db_norm[idx]:>+16.2f}")


if __name__ == "__main__":
    main()
