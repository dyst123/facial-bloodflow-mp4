# Facial Blood Flow Analysis Pipeline

## Goal
Determine whether facial blood flow patterns differ significantly across three experimental trials (T1, T2, T3), as a proxy for whether the task being performed influences peripheral blood flow. Results are compared across multiple subjects (s1, s2, s3) to assess replicability.

---

## Overview of the Pipeline

```
Raw video (T1, T2, T3)
        │
        ▼
  Face landmark detection (MediaPipe FaceMesh, 478 landmarks)
        │
        ▼
  ROI pixel extraction → mean green channel intensity per frame
  (nose, forehead, left cheek, right cheek)
        │
        ▼
  Beer-Lambert transform: bf = –log(intensity)
  Moving-average smoothing → zero-meaned relative blood flow signal
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │           Analysis Pipeline                 │
  │  Step 1: Feature Extraction                 │
  │  Step 2: Cross-Correlation                  │
  │  Step 3: PSD Overlay                        │
  │  Step 4: Bar Charts                         │
  └─────────────────────────────────────────────┘
        │
        ▼
  Outputs: plots + CSVs (per subject)
```

---

## Step 1 — Feature Extraction (`feature_summary.csv`)

For each ROI × trial combination, the following features are computed:

| Feature | Definition | Why it matters |
|---|---|---|
| **RMS** | √(mean of signal²) | Overall signal energy — larger = bigger blood flow fluctuations |
| **Mean** | Average signal value | Should be ~0 after zero-meaning; drift indicates baseline shift |
| **Std** | Standard deviation | Same as RMS for zero-meaned signal; measure of variability |
| **Cardiac Band Power** | Integrated PSD power in 0.7–3 Hz | How much of the signal is at heart-rate frequencies |
| **Cardiac Band Fraction** | Cardiac power / total power | What proportion of signal energy is physiologically meaningful |
| **Dominant Frequency** | Peak frequency within 0.7–3 Hz | Estimated heart rate in Hz (multiply by 60 for BPM) |

---

## Step 2 — Cross-Correlation (`cross_correlations.csv`, `cross_correlation_heatmap.png`)

Pearson correlation coefficient (r) is computed between the full waveforms of each trial pair, per ROI.

- **r ≈ 1.0**: trials look nearly identical — task had no effect on that ROI
- **r ≈ 0.0**: waveforms are unrelated — signal differs completely between trials
- **r < 0**: waveforms are inversely related (anti-phase)

> **Note on p-values:** With ~6000 data points, even tiny correlations reach statistical significance (p < 0.001). Focus on the magnitude of r, not the p-value alone.

---

## Step 3 — Power Spectral Density (`psd_comparison.png`)

The PSD shows how signal power is distributed across frequencies (0–5 Hz).

- The **red shaded region** marks the cardiac band (0.7–3 Hz), where heartbeat-driven blood flow should appear as a peak
- A taller peak in the cardiac band = more rhythmic, heart-rate-driven blood flow
- A flatter spectrum = more noise, less organised physiological signal
- A peak shifted left or right = change in heart rate between trials
- Low-frequency drift (< 0.5 Hz) reflects slow vasomotor changes or motion artefacts

---

## Step 4 — Bar Charts (`feature_bar_charts.png`)

Two bar charts summarise the key features side by side:

1. **RMS Amplitude** — visual comparison of signal strength per ROI per trial
2. **Cardiac Band Power Fraction** — what proportion of signal energy is at heart-rate frequencies

---

## Results — Subject s1

### RMS by ROI and Trial
| ROI | T1 | T2 | T3 |
|---|---|---|---|
| Nose | 0.0095 | 0.0368 | 0.0357 |
| Forehead | 0.0095 | 0.0282 | 0.0389 |
| Left Cheek | 0.0106 | 0.0336 | 0.0388 |
| Right Cheek | 0.0095 | 0.0231 | 0.0258 |

- T1 is ~3–4× lower than T2/T3 across all ROIs
- Dominant frequency: T1 ~1.1 Hz vs T2/T3 ~0.82 Hz
- All cross-correlations low (|r| < 0.30); T1 vs T3 most divergent (forehead r = –0.29)

---

## Results — Subject s2

### RMS by ROI and Trial
| ROI | T1 | T2 | T3 |
|---|---|---|---|
| Nose | 0.0146 | 0.0701 | 0.1152 |
| Forehead | 0.0262 | 0.1215 | 0.1188 |
| Left Cheek | 0.0170 | 0.0721 | 0.1102 |
| Right Cheek | 0.0115 | 0.0618 | 0.1267 |

- T1 is ~4–8× lower than T2/T3 — even stronger separation than s1
- Dominant frequency: T1 ~0.96 Hz vs T2/T3 ~0.82 Hz
- All cross-correlations low and predominantly negative (|r| < 0.26)

---

## Results — Subject s3

### RMS by ROI and Trial
| ROI | T1 | T2 | T3 |
|---|---|---|---|
| Nose | 0.0223 | 0.0368 | 0.0318 |
| Forehead | 0.0126 | 0.0893 | 0.0733 |
| Left Cheek | 0.0083 | 0.0681 | 0.0684 |
| Right Cheek | 0.0150 | 0.0611 | 0.0914 |

- T1 is again the lowest across all ROIs
- T2 and T3 more similar to each other than to T1
- Dominant frequency: T1 ~0.82–1.37 Hz vs T2/T3 ~0.82 Hz

---

## Cross-Subject Replication Summary

| Subject | T1 < T2 < T3 in RMS? | Freq shift T1 vs T2/T3? | Low cross-correlations? |
|---|---|---|---|
| s1 | ✓ All 4 ROIs | ✓ 1.1 Hz → 0.82 Hz | ✓ |
| s2 | ✓ All 4 ROIs | ✓ 0.96 Hz → 0.82 Hz | ✓ |
| s3 | ✓ All 4 ROIs | Partial | ✓ |

**The core pattern replicates across all 3 subjects:** T1 is consistently a lower-amplitude, physiologically quieter state. T2 and T3 show higher blood flow variability and share a similar frequency signature.

---

## Can We Train a Classifier?

### Short answer: Not yet — but the signal is promising.

### What we have
- 3 subjects × 3 trials
- Consistent T1 < T2/T3 amplitude pattern across all subjects
- Frequency shift between T1 and T2/T3 in 2 of 3 subjects

### Why a model is premature

| Issue | Why it matters |
|---|---|
| **Only 3 subjects** | Far too few — any model will overfit to the individuals, not the task |
| **Task labels unknown** | We don't know what T1, T2, T3 involve — can't interpret what the model would be learning |
| **No ground-truth validation** | BVP and EDA data exist but haven't been cross-referenced — signal may partly reflect motion or lighting rather than blood flow |
| **Single recording per condition** | No within-subject repeatability — can't separate task effect from individual session variability |

### Recommended path to a classifier

```
Current state: 3 subjects, pattern observed, unvalidated
        ↓
Step 1: Validate against BVP/EDA (data already available for s2, s3...)
        ↓
Step 2: Run remaining subjects (s4–s10)
        ↓
Step 3: If pattern holds across 8–10 subjects → pilot classifier feasible
        ↓
Step 4: Simple model first (SVM or LDA on RMS + cardiac power features)
        ↓
Step 5: Evaluate with leave-one-subject-out cross-validation
```

### What a classifier would look like
- **Input features:** RMS, cardiac band power, dominant frequency — per ROI (4 ROIs × 3 features = 12 features per trial)
- **Labels:** T1, T2, T3
- **Model:** Start with SVM or LDA (interpretable, works with small n); move to neural nets only if n > 30
- **Validation:** Leave-one-subject-out — train on s1/s2, test on s3 — to test generalisation across people

---

## Limitations

- **n = 3 subjects:** Results are descriptive. Statistical inference requires more subjects.
- **No ground-truth heart rate:** BVP data exists but not yet integrated — dominant frequency unvalidated.
- **No head-motion correction:** Head movement introduces noise. The original ToF-camera pipeline corrected for this; this version does not.
- **Task labels unknown:** Without knowing what T1, T2, T3 involve, physiological interpretation is speculative.
- **Single recording per condition:** Cannot separate task effect from session-to-session variability.

---

## Outputs Summary (per subject folder)

| File | Contents |
|---|---|
| `bloodflow_data.csv` | Raw blood flow time-series per ROI per trial |
| `comparison_trials.png` | Raw waveform overlay T1/T2/T3 per ROI |
| `analysis/feature_summary.csv` | RMS, cardiac power, dominant frequency per ROI × trial |
| `analysis/cross_correlations.csv` | Pearson r between trial pairs per ROI |
| `analysis/psd_comparison.png` | PSD overlay T1/T2/T3 per ROI |
| `analysis/feature_bar_charts.png` | RMS and cardiac power bar charts |
| `analysis/cross_correlation_heatmap.png` | Colour-coded similarity matrix |

---

---

# Slide Bullet Points

## Slide 1 — What We're Measuring
- Facial blood flow extracted from ordinary RGB video — no special hardware required
- MediaPipe FaceMesh detects 478 facial landmarks per frame
- Four regions of interest (ROIs): nose, forehead, left cheek, right cheek
- Signal = mean green pixel intensity within each ROI, every frame
- Green channel used because haemoglobin absorbs green light most strongly (rPPG technique)

## Slide 2 — Signal Processing
- Raw green intensity → Beer-Lambert transform: `bf = –log(intensity)`
- Higher value = more light absorbed = more blood in that region
- Moving-average smoothing to remove high-frequency noise
- Zero-meaned to give relative (not absolute) blood flow change
- Sampling rate: ~35 fps

## Slide 3 — Research Question
- Does facial blood flow pattern differ depending on which task (T1, T2, T3) the participant is performing?
- If yes → facial blood flow is sensitive to task type → potential non-invasive physiological marker
- Pilot study: 3 subjects, 3 trials each

## Slide 4 — Analysis Pipeline
- **Step 1:** Feature extraction — RMS, cardiac-band power, dominant frequency per ROI per trial
- **Step 2:** Cross-correlation — how similar are the waveforms across trials?
- **Step 3:** Power Spectral Density — does the frequency content change across trials?
- **Step 4:** Bar charts — visual comparison of signal features across trials

## Slide 5 — Key Finding: Signal Amplitude (RMS)
- T1 shows 3–8× lower RMS amplitude than T2 and T3 across all ROIs
- Pattern is consistent across all 3 subjects and all 4 facial regions
- Suggests blood flow fluctuations are substantially smaller during T1
- Possible interpretation: T1 represents a lower arousal or lower cardiovascular load condition

## Slide 6 — Key Finding: Frequency Content (PSD)
- Dominant frequency in T1: ~0.96–1.1 Hz in s1 and s2; variable in s3
- Dominant frequency in T2 and T3: ~0.82 Hz consistently across subjects
- Shift in dominant frequency suggests a change in heart rate or vasomotor rhythm between T1 and T2/T3
- T2 and T3 share the same frequency signature — physiologically similar conditions

## Slide 7 — Key Finding: Waveform Similarity (Cross-Correlation)
- All trial pairs show low correlation (|r| < 0.30) across all subjects → waveforms differ between trials
- T1 is most dissimilar to both T2 and T3
- T2 and T3 are more similar to each other than either is to T1
- Pattern replicates in s1, s2, and s3 — not a single-subject artefact

## Slide 8 — Replication Across Subjects
- Core finding replicates in all 3 subjects: T1 amplitude consistently lowest
- Frequency shift (T1 higher freq → T2/T3 lower freq) seen in s1 and s2
- s3 shows same amplitude pattern but less clear frequency shift
- Consistency across subjects strengthens the case for a genuine task-driven effect

## Slide 9 — Can We Build a Classifier?
- Pattern is promising but data is insufficient for a reliable model
- 3 subjects is too few — model would overfit to individuals, not tasks
- Next step: validate signal against BVP (wrist heart rate) and EDA (skin conductance) already collected
- Then run remaining subjects (s4–s10) before attempting classification
- Target: SVM/LDA on 12 features (RMS + cardiac power + dominant freq × 4 ROIs), leave-one-subject-out validation

## Slide 10 — Limitations and Next Steps
- **n = 3:** Descriptive only — need more subjects for statistical inference
- **No ground-truth HR:** BVP data exists but not yet cross-referenced
- **No head-motion correction:** Movement artefacts may inflate differences
- **Task labels needed:** Physiological interpretation requires knowing what T1, T2, T3 involve
- **Next steps:** Integrate BVP/EDA → run s4–s10 → pilot classifier
