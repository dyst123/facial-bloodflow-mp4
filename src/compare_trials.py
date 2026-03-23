"""
compare_trials.py
-----------------
Loads the per-trial blood flow data and plots a horizontal comparison:
  rows = ROI regions, columns = trials (T1, T2, T3)

Usage:
    python compare_trials.py s1                        # subject s1 from default Data dir
    python compare_trials.py s2 D:/Downloads/s1_to_s10/s2   # subject s2 from custom dir
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from facemeshmodule_new import FaceMeshDetector

FPS        = 30
SMOOTH_WIN = 20
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'Data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

ROI_DEFINITIONS = {
    'nose':        [196, 419, 455, 235],
    'forehead':    [109, 338, 9],
    'left_cheek':  [129, 121, 117, 216],
    'right_cheek': [436, 346, 343, 344],
}

TRIAL_LABELS = ['T1', 'T2', 'T3']


def get_roi_mask(landmark_indices, landmarks, h, w):
    points = [tuple(landmarks[i]) for i in landmark_indices]
    canvas = Image.new('L', (w, h), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=1, outline=1)
    return np.array(canvas, dtype=bool)


def mean_green(frame_rgb, mask):
    green = frame_rgb[:, :, 1]
    pixels = green[mask]
    return float(np.mean(pixels)) if pixels.size > 0 else 0.0


def smooth(x, window=SMOOTH_WIN):
    if len(x) <= window:
        return x
    s = np.r_[x[window-1:0:-1], x, x[-2:-window-1:-1]]
    y = np.convolve(np.ones(window) / window, s, mode='valid')
    return y[window//2: window//2 + len(x)]


def relative_bloodflow(intensity_series):
    intensity_series = np.array(intensity_series, dtype=float)
    intensity_series = np.clip(intensity_series, 1e-6, None)
    bf = -np.log(intensity_series)
    bf = smooth(bf)
    bf -= np.mean(bf)
    return bf


def extract_bloodflow(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return None, None
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    detector = FaceMeshDetector(static_image_mode=False, max_num_faces=1)
    series = {roi: [] for roi in ROI_DEFINITIONS}

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        face_detected, landmarks = detector.find_face_mesh(frame_bgr, draw=False)
        if not face_detected:
            for roi in ROI_DEFINITIONS:
                series[roi].append(series[roi][-1] if series[roi] else 0.0)
            continue
        for roi, indices in ROI_DEFINITIONS.items():
            mask = get_roi_mask(indices, landmarks, h, w)
            series[roi].append(mean_green(frame_rgb, mask))

    cap.release()
    bf = {roi: relative_bloodflow(series[roi]) for roi in ROI_DEFINITIONS}
    time_axis = np.arange(len(bf['nose'])) / actual_fps
    return bf, time_axis


def main():
    # Parse subject ID and optional video directory from command line
    subject = sys.argv[1] if len(sys.argv) > 1 else 's1'
    video_dir = sys.argv[2] if len(sys.argv) > 2 else None

    search_dirs = []
    if video_dir:
        search_dirs.append(video_dir)
    search_dirs += ['d:/facial-bloodflow-mp4-main/Data', DATA_DIR]

    video_files = [f'vid_{subject}_T1.avi', f'vid_{subject}_T2.avi', f'vid_{subject}_T3.avi']

    all_bf = []
    all_time = []
    found_labels = []

    for fname, label in zip(video_files, TRIAL_LABELS):
        path = None
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.isfile(candidate):
                path = candidate
                break
        if path is None:
            print(f"Skipping {fname} — not found")
            continue
        print(f"Processing {label}: {path}")
        bf, time_axis = extract_bloodflow(path)
        if bf is None:
            continue
        all_bf.append(bf)
        all_time.append(time_axis)
        found_labels.append(label)

    if not all_bf:
        print("No videos processed.")
        return

    rois = list(ROI_DEFINITIONS.keys())
    n_rows = len(rois)
    colors = ['steelblue', 'darkorange', 'seagreen']

    # ── Save CSV data ──────────────────────────────────────────────────────────
    import csv
    subject_out = os.path.join(OUTPUT_DIR, subject)
    os.makedirs(subject_out, exist_ok=True)
    csv_path = os.path.join(subject_out, 'bloodflow_data.csv')

    # Find max length across all trials/rois
    max_len = max(len(bf[roi]) for bf in all_bf for roi in rois)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: time_T1, time_T2, time_T3, nose_T1, nose_T2, ...
        header = []
        for label, time_axis in zip(found_labels, all_time):
            header.append(f'time_{label}_s')
        for roi in rois:
            for label in found_labels:
                header.append(f'{roi}_{label}')
        writer.writerow(header)

        for i in range(max_len):
            row_data = []
            for time_axis in all_time:
                row_data.append(f'{time_axis[i]:.4f}' if i < len(time_axis) else '')
            for roi in rois:
                for bf in all_bf:
                    row_data.append(f'{bf[roi][i]:.6f}' if i < len(bf[roi]) else '')
            writer.writerow(row_data)

    print(f"Saved data: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), sharey=True)
    if n_rows == 1:
        axes = [axes]

    for row, roi in enumerate(rois):
        ax = axes[row]
        for col, (label, bf, time_axis) in enumerate(zip(found_labels, all_bf, all_time)):
            ax.plot(time_axis[:len(bf[roi])], bf[roi],
                    color=colors[col % len(colors)], linewidth=0.8, label=label, alpha=0.85)
        ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
        ax.grid(True, alpha=0.4)
        ax.set_title(roi.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Rel. Blood Flow (a.u.)', fontsize=9)
        ax.legend(fontsize=9, loc='upper right')
        if row == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=10)

    fig.suptitle(
        f'{subject.upper()} — Blood Flow by Region — T1 vs T2 vs T3\n'
        'Y-axis: –log(mean green intensity), zero-meaned (arbitrary units)\n'
        'Higher = more green light absorbed = more blood in ROI',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    out_path = os.path.join(subject_out, 'comparison_trials.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot:  {out_path}")


if __name__ == '__main__':
    main()
