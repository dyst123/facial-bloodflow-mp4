"""
analyse_trials.py
-----------------
Analysis pipeline for comparing facial blood flow across trials T1, T2, T3.

Steps:
  1. Load saved bloodflow_data.csv
  2. Extract features per ROI per trial (RMS, mean, dominant frequency, cardiac-band power)
  3. Cross-correlation matrix between trials (per ROI)
  4. PSD overlay plot (T1/T2/T3 per ROI)
  5. Bar charts of RMS and cardiac-band power
  6. Save feature summary table as CSV

Usage:
    python analyse_trials.py           # defaults to s1
    python analyse_trials.py s2        # run for subject s2
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sci_signal
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────
SUBJECT     = sys.argv[1] if len(sys.argv) > 1 else 's1'
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
CSV_PATH    = os.path.join(OUTPUT_DIR, SUBJECT, 'bloodflow_data.csv')
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, SUBJECT, 'analysis')

# ── Constants ──────────────────────────────────────────────────────────────────
FS = 35.1                        # sampling rate (fps of the videos)
CARDIAC_BAND = (0.7, 3.0)        # Hz — typical resting heart rate range
ROIS    = ['nose', 'forehead', 'left_cheek', 'right_cheek']
TRIALS  = ['T1', 'T2', 'T3']
COLORS  = {'T1': 'steelblue', 'T2': 'darkorange', 'T3': 'seagreen'}

os.makedirs(ANALYSIS_DIR, exist_ok=True)


# ── 1. Load data ───────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(CSV_PATH)
    data = {}   # data[roi][trial] = np.array of signal
    for roi in ROIS:
        data[roi] = {}
        for trial in TRIALS:
            col = f'{roi}_{trial}'
            series = df[col].dropna().values.astype(float)
            data[roi][trial] = series
    return data


# ── 2. Feature extraction ──────────────────────────────────────────────────────

def cardiac_band_power(sig, fs=FS, band=CARDIAC_BAND):
    """Fraction of total power within the cardiac frequency band."""
    freqs, psd = sci_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    total_power = np.trapezoid(psd, freqs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.trapezoid(psd[mask], freqs[mask])
    return band_power, total_power, freqs, psd


def dominant_frequency(freqs, psd, band=CARDIAC_BAND):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan
    return freqs[mask][np.argmax(psd[mask])]


def extract_features(data):
    rows = []
    psd_store = {}   # psd_store[roi][trial] = (freqs, psd)

    for roi in ROIS:
        psd_store[roi] = {}
        for trial in TRIALS:
            sig = data[roi][trial]
            rms = np.sqrt(np.mean(sig ** 2))
            mean_val = np.mean(sig)
            std_val = np.std(sig)
            band_pow, total_pow, freqs, psd = cardiac_band_power(sig)
            dom_freq = dominant_frequency(freqs, psd)
            cardiac_frac = band_pow / total_pow if total_pow > 0 else 0

            psd_store[roi][trial] = (freqs, psd)
            rows.append({
                'ROI':                    roi,
                'Trial':                  trial,
                'RMS':                    round(rms, 6),
                'Mean':                   round(mean_val, 6),
                'Std':                    round(std_val, 6),
                'Cardiac_Band_Power':     round(band_pow, 8),
                'Cardiac_Band_Fraction':  round(cardiac_frac, 4),
                'Dominant_Freq_Hz':       round(dom_freq, 3),
            })

    features_df = pd.DataFrame(rows)
    return features_df, psd_store


# ── 3. Cross-correlation ───────────────────────────────────────────────────────

def compute_cross_correlations(data):
    pairs = [('T1', 'T2'), ('T1', 'T3'), ('T2', 'T3')]
    rows = []
    for roi in ROIS:
        for (ta, tb) in pairs:
            sig_a = data[roi][ta]
            sig_b = data[roi][tb]
            n = min(len(sig_a), len(sig_b))
            r, p = pearsonr(sig_a[:n], sig_b[:n])
            rows.append({'ROI': roi, 'Pair': f'{ta} vs {tb}',
                         'Pearson_r': round(r, 4), 'p_value': round(p, 6)})
    return pd.DataFrame(rows)


# ── 4. PSD overlay plot ────────────────────────────────────────────────────────

def plot_psd(psd_store):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for ax, roi in zip(axes, ROIS):
        for trial in TRIALS:
            freqs, psd = psd_store[roi][trial]
            ax.semilogy(freqs, psd, color=COLORS[trial], linewidth=1.2, label=trial, alpha=0.85)
        ax.axvspan(*CARDIAC_BAND, alpha=0.08, color='red', label='Cardiac band' if roi == ROIS[0] else '')
        ax.set_ylabel('PSD (a.u.²/Hz)', fontsize=9)
        ax.set_title(roi.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)
    axes[-1].set_xlabel('Frequency (Hz)', fontsize=10)
    fig.suptitle('Power Spectral Density by ROI — T1 vs T2 vs T3\n'
                 '(red band = cardiac range 0.7–3 Hz)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(ANALYSIS_DIR, 'psd_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 5. Bar charts ──────────────────────────────────────────────────────────────

def plot_bar_charts(features_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = [
        ('RMS',                   'RMS Amplitude (a.u.)',            'RMS of Blood Flow Signal'),
        ('Cardiac_Band_Fraction', 'Cardiac Band Power Fraction',     'Cardiac Band Power (fraction of total)'),
    ]

    x = np.arange(len(ROIS))
    width = 0.25

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for i, trial in enumerate(TRIALS):
            vals = [features_df.loc[(features_df.ROI == roi) & (features_df.Trial == trial), metric].values[0]
                    for roi in ROIS]
            ax.bar(x + i * width, vals, width, label=trial, color=COLORS[trial], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels([r.replace('_', '\n') for r in ROIS], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle('Signal Features by ROI and Trial', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(ANALYSIS_DIR, 'feature_bar_charts.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 6. Cross-correlation heatmap ───────────────────────────────────────────────

def plot_correlation_heatmap(xcorr_df):
    pairs = xcorr_df['Pair'].unique()
    matrix = np.zeros((len(ROIS), len(pairs)))

    for j, pair in enumerate(pairs):
        for i, roi in enumerate(ROIS):
            r = xcorr_df.loc[(xcorr_df.ROI == roi) & (xcorr_df.Pair == pair), 'Pearson_r'].values[0]
            matrix[i, j] = r

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=10)
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels([r.replace('_', ' ').title() for r in ROIS], fontsize=10)
    for i in range(len(ROIS)):
        for j in range(len(pairs)):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=11,
                    color='black')
    ax.set_title('Cross-Correlation Between Trials (Pearson r)\n'
                 'Green = similar waveforms, Red = different', fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(ANALYSIS_DIR, 'cross_correlation_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_data()

    print("Extracting features...")
    features_df, psd_store = extract_features(data)
    feat_path = os.path.join(ANALYSIS_DIR, 'feature_summary.csv')
    features_df.to_csv(feat_path, index=False)
    print(f"Saved: {feat_path}")
    print("\nFeature Summary:")
    print(features_df.to_string(index=False))

    print("\nComputing cross-correlations...")
    xcorr_df = compute_cross_correlations(data)
    xcorr_path = os.path.join(ANALYSIS_DIR, 'cross_correlations.csv')
    xcorr_df.to_csv(xcorr_path, index=False)
    print(f"Saved: {xcorr_path}")
    print("\nCross-Correlations:")
    print(xcorr_df.to_string(index=False))

    print("\nGenerating plots...")
    plot_psd(psd_store)
    plot_bar_charts(features_df)
    plot_correlation_heatmap(xcorr_df)

    print("\nDone. All outputs saved to:", ANALYSIS_DIR)


if __name__ == '__main__':
    main()
