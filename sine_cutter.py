import argparse
import os
import shutil
import subprocess
import sys

import numpy as np


def decode_stream(path, sr):
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-ar", str(sr), "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def read_samples(proc, n):
    raw = proc.stdout.read(n * 2)
    got = len(raw) // 2
    if got < n:
        return None, got
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, got


def detect(path, sr, block, hop, zp, fmin, fmax, snr_db, weak_db, floor_db,
           silence_rms, track, ftol, break_frames, gap, min_len, pad):
    nfft = block * zp
    proc = decode_stream(path, sr)
    win = np.hanning(block).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() < 16:
        proc.stdout.close()
        proc.wait()
        raise SystemExit("frequency band too narrow for this block size")
    fb = freqs[mask]
    nb = max(2, int(ftol * nfft / sr))
    amp_ref = block / 4.0

    buf = np.zeros(block, dtype=np.float32)
    first, got = read_samples(proc, block)
    if first is None:
        proc.stdout.close()
        rc = proc.wait()
        if rc == 0:
            print("warning: audio shorter than one analysis window "
                  "(%d samples needed)" % block, file=sys.stderr)
        else:
            print("warning: ffmpeg failed to decode input (exit %d)" % rc,
                  file=sys.stderr)
        return [], got / sr, sr
    buf[:] = first

    hist = []

    def frame(x):
        rms = float(np.sqrt(np.mean(x * x)))
        if rms < silence_rms:
            return None
        spec = np.abs(np.fft.rfft(x * win, nfft))
        band = spec[mask]
        i = int(np.argmax(band))
        peak = float(band[i])
        w = max(64, 4 * nb)
        lo2 = max(0, i - nb - w)
        hi2 = min(band.size, i + nb + w)
        ilo = max(0, i - nb)
        ihi = min(band.size, i + nb + 1)
        rest = np.concatenate((band[lo2:ilo], band[ihi:hi2]))
        if rest.size < 32:
            rest = np.concatenate((band[:ilo], band[ihi:]))
        bg = float(np.median(rest)) + 1e-12
        snr = 20.0 * np.log10(peak / bg) if peak > 0.0 else -99.0
        amp = 20.0 * np.log10(peak / amp_ref + 1e-12)
        edge = (i < nb) or (i >= band.size - nb)
        return float(fb[i]), snr, amp, edge

    def push(f, s, e):
        hist.append((f, s, e))
        if len(hist) > track:
            del hist[0]

    def has_sine(f, s, amp, edge):
        if amp < floor_db or edge:
            return False
        stable = []
        for hf, hs, he in hist:
            if he or abs(hf - f) > ftol:
                return False
            stable.append(hs)
        if len(stable) < track:
            return False
        if s > snr_db:
            return True
        for hs in stable:
            if hs <= weak_db:
                return False
        return True

    off = block - 0.5 * hop
    hits = []
    total = float(got)
    idx = 0

    def hit():
        hits.append(idx)

    r = frame(buf)
    if r is None:
        push(-1.0, -99.0, True)
    else:
        push(r[0], r[1], r[3])
        if has_sine(r[0], r[1], r[2], r[3]):
            hit()

    while True:
        new, got = read_samples(proc, hop)
        total += float(got)          # 尾部不足一个 hop 的残样也要计入时长
        if new is None:
            break
        buf[:-hop] = buf[hop:]
        buf[-hop:] = new
        idx += 1
        r = frame(buf)
        if r is None:
            push(-1.0, -99.0, True)
            continue
        push(r[0], r[1], r[3])
        if has_sine(r[0], r[1], r[2], r[3]):
            hit()

    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        print("warning: ffmpeg exited with code %d, result may be truncated"
              % rc, file=sys.stderr)

    duration = total / sr
    runs = []
    if hits:
        s0 = hits[0]
        p0 = hits[0]
        for h in hits[1:]:
            if h - p0 <= break_frames:
                p0 = h
            else:
                runs.append((s0, p0))
                s0 = h
                p0 = h
        runs.append((s0, p0))

    lag = (track - 1) * hop
    intervals = []
    for i0, i1 in runs:
        a = 0.0 if i0 == 0 else (i0 * hop + off - lag) / sr
        if i1 == idx:          
            b = duration
        else:                  
            b = min(duration, (i1 * hop + 0.5 * hop) / sr)
        if b - a < min_len:
            continue
        intervals.append([a, b])

    merged = []
    for a, b in intervals:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    out = []
    prev_end = None
    for a, b in merged:
        a2 = max(0.0, a - pad)
        b2 = min(duration, b + pad)
        if prev_end is not None and a2 < prev_end:
            a2 = prev_end
        if b2 - a2 <= 0.0:
            continue
        out.append((a2, b2))
        prev_end = b2
    return out, duration, sr


def cut(path, seg, outdir, index):
    a, b = seg
    name = "clip_%04d_%08.2f_%08.2f.wav" % (index, a, b)
    out = os.path.join(outdir, name)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-ss", "%.3f" % a,
        "-i", path,
        "-t", "%.3f" % (b - a),
        "-c:a", "pcm_s16le",
        out,
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        return None
    return out


def fmt(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return "%02d:%02d:%06.3f" % (h, m, s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--outdir", default="sine_clips")
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--zp", type=int, default=2)
    p.add_argument("--fmin", type=float, default=100.0)
    p.add_argument("--fmax", type=float, default=3500.0)
    p.add_argument("--snr", type=float, default=12.0)
    p.add_argument("--weak", type=float, default=4.0)
    p.add_argument("--floor", type=float, default=-95.0)
    p.add_argument("--silence", type=float, default=0.00001)
    p.add_argument("--track", type=int, default=14)
    p.add_argument("--breakf", type=int, default=4)
    p.add_argument("--ftol", type=float, default=3.0)
    p.add_argument("--gap", type=float, default=0.5,
                   help="merge segments whose gap is <= this many seconds "
                        "(before padding)")
    p.add_argument("--minlen", type=float, default=0.1)
    p.add_argument("--pad", type=float, default=0.5)
    p.add_argument("--no-cut", action="store_true")
    a = p.parse_args()

    if not os.path.isfile(a.input):
        print("input not found: %s" % a.input)
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH")
        return 1

    if a.sr <= 0:
        raise SystemExit("--sr must be > 0")
    if a.block <= 0:
        raise SystemExit("--block must be > 0")
    if a.hop <= 0:
        raise SystemExit("--hop must be > 0")
    if a.hop > a.block:
        raise SystemExit("--hop must be <= --block")
    if a.zp < 1:
        raise SystemExit("--zp must be >= 1")
    if a.track < 1:
        raise SystemExit("--track must be >= 1")
    if a.fmin >= a.fmax:
        raise SystemExit("--fmin must be < --fmax")
    if a.fmax > a.sr / 2.0:
        raise SystemExit("--fmax exceeds nyquist (sr/2 = %.1f)" % (a.sr / 2.0))
    if a.pad < 0:
        raise SystemExit("--pad must be >= 0")

    os.makedirs(a.outdir, exist_ok=True)

    segs, dur, sr = detect(
        a.input, a.sr, a.block, a.hop, a.zp, a.fmin, a.fmax,
        a.snr, a.weak, a.floor, a.silence, a.track, a.ftol,
        a.breakf, a.gap, a.minlen, a.pad,
    )

    print("duration: %s" % fmt(dur))
    print("segments: %d" % len(segs))

    lines = []
    for i, (x, y) in enumerate(segs, 1):
        lines.append("%04d,%s,%s,%.3f,%.3f,%.3f" % (i, fmt(x), fmt(y), x, y, y - x))
    with open(os.path.join(a.outdir, "segments.csv"), "w") as f:
        f.write("index,start,end,start_sec,end_sec,duration_sec\n")
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    if a.no_cut:
        print("written: %s" % os.path.join(a.outdir, "segments.csv"))
        return 0

    ok = 0
    for i, (x, y) in enumerate(segs, 1):
        r = cut(a.input, (x, y), a.outdir, i)
        if r:
            ok += 1
        print("[%d/%d] %s -> %s" % (i, len(segs), fmt(x), os.path.basename(r) if r else "FAILED"))

    print("saved clips: %d" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
