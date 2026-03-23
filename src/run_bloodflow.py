"""
run_bloodflow.py
----------------
Input:  MP4 video file (put in ../Data/)
Output: Relative blood flow plots saved to ../outputs/<video_name>/

Usage:
    python run_bloodflow.py                  # processes all MP4s in ../Data/
    python run_bloodflow.py myvideo.mp4      # processes one specific file
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from facemeshmodule_new import FaceMeshDetector

# ── Settings ──────────────────────────────────────────────────────────────────
FPS         = 30        # expected FPS of your video
SMOOTH_WIN  = 20        # smoothing window for blood flow signal (frames)
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'Data')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# MediaPipe landmark indices that define each ROI polygon
ROI_DEFINITIONS = {
    'nose':        [196, 419, 455, 235],
    'forehead':    [109, 338, 9],
    'left_cheek':  [129, 121, 117, 216],
    'right_cheek': [436, 346, 343, 344],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_roi_mask(landmark_indices, landmarks, h, w):
    """Return a binary mask (h x w) for the polygon defined by landmark_indices."""
    points = [tuple(landmarks[i]) for i in landmark_indices]
    canvas = Image.new('L', (w, h), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=1, outline=1)
    return np.array(canvas, dtype=bool)


def mean_green(frame_rgb, mask):
    """Mean green channel value inside mask."""
    green = frame_rgb[:, :, 1]
    pixels = green[mask]
    return float(np.mean(pixels)) if pixels.size > 0 else 0.0


def smooth(x, window=SMOOTH_WIN):
    """Moving-average smoothing."""
    if len(x) <= window:
        return x
    s = np.r_[x[window-1:0:-1], x, x[-2:-window-1:-1]]
    y = np.convolve(np.ones(window) / window, s, mode='valid')
    return y[window//2: window//2 + len(x)]


def relative_bloodflow(intensity_series):
    """
    Convert raw intensity time-series to relative blood flow using Beer-Lambert:
        bf = -log(I) then smooth
    A drop in reflected green light → increase in blood absorption → positive bf spike.
    """
    intensity_series = np.array(intensity_series, dtype=float)
    intensity_series = np.clip(intensity_series, 1e-6, None)   # avoid log(0)
    bf = -np.log(intensity_series)
    bf = smooth(bf)
    bf -= np.mean(bf)   # zero-mean (relative)
    return bf


# ── Core processing ───────────────────────────────────────────────────────────

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps   = cap.get(cv2.CAP_PROP_FPS) or FPS
    print(f"  {total_frames} frames @ {actual_fps:.1f} fps")

    detector = FaceMeshDetector(static_image_mode=False, max_num_faces=1)

    # Per-frame accumulators
    series = {roi: [] for roi in ROI_DEFINITIONS}
    skipped = 0

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        face_detected, landmarks = detector.find_face_mesh(frame_bgr, draw=False)
        if not face_detected:
            # fill with previous value or 0
            for roi in ROI_DEFINITIONS:
                series[roi].append(series[roi][-1] if series[roi] else 0.0)
            skipped += 1
            frame_idx += 1
            continue

        for roi, indices in ROI_DEFINITIONS.items():
            mask = get_roi_mask(indices, landmarks, h, w)
            series[roi].append(mean_green(frame_rgb, mask))

        frame_idx += 1

    cap.release()
    print(f"  {skipped}/{frame_idx} frames had no face detected")

    # ── Compute relative blood flow for each ROI ───────────────────────────
    bf = {}
    for roi in ROI_DEFINITIONS:
        bf[roi] = relative_bloodflow(series[roi])

    time_axis = np.arange(len(bf['nose'])) / actual_fps

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]

    # ── Plot 1: Individual ROI blood flow ─────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, roi in zip(axes, ROI_DEFINITIONS):
        ax.plot(time_axis[:len(bf[roi])], bf[roi])
        ax.set_ylabel('Rel. BF (a.u.)')
        ax.set_title(f'{name} — {roi.replace("_", " ").title()} Blood Flow')
        ax.grid(True)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    out1 = os.path.join(output_dir, f'bloodflow_all_rois_{name}.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out1}")

    # ── Plot 2: Nose − Forehead (matches the style in the repo's paper) ───
    n = min(len(bf['nose']), len(bf['forehead']))
    bf_diff = bf['nose'][:n] - bf['forehead'][:n]

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis[:n], bf_diff, color='steelblue', label='Nose − Forehead')
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Blood Flow (a.u.)')
    plt.title(f'{name} Forehead Bloodflow')
    plt.legend()
    plt.grid(True)
    out2 = os.path.join(output_dir, f'bloodflow_nose_forehead_{name}.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out2}")

    # ── Plot 3: Forehead only (matches the Alex graph style) ──────────────
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis[:len(bf['forehead'])], bf['forehead'], color='steelblue', label='forehead')
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Blood Flow (a.u.)')
    plt.title(f'{name} Forehead Bloodflow')
    plt.legend()
    plt.grid(True)
    out3 = os.path.join(output_dir, f'bloodflow_forehead_{name}.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out3}")

    return bf, time_axis


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        # specific file passed as argument
        video_path = sys.argv[1]
        if not os.path.isabs(video_path):
            video_path = os.path.join(DATA_DIR, video_path)
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(OUTPUT_DIR, name)
        print(f"Processing: {video_path}")
        process_video(video_path, out_dir)
    else:
        # process all MP4s in Data/
        mp4_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not mp4_files:
            print(f"No MP4 files found in {DATA_DIR}")
            return
        for f in mp4_files:
            video_path = os.path.join(DATA_DIR, f)
            name = os.path.splitext(f)[0]
            out_dir = os.path.join(OUTPUT_DIR, name)
            print(f"\nProcessing: {f}")
            process_video(video_path, out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
