# Facial Blood Flow Analysis — Key Takeaways

## What We Did
We extracted facial blood flow signals from ordinary RGB video recordings of subjects performing three different tasks (T1, T2, T3). Using MediaPipe FaceMesh to track 478 facial landmarks per frame, we measured the mean green channel intensity within four facial regions (nose, forehead, left cheek, right cheek) and converted these into relative blood flow signals using the Beer-Lambert relationship. We then applied a feature-based analysis pipeline to compare the signals across trials and across three subjects (s1, s2, s3).

---

## Core Findings

### 1. T1 is consistently a quieter physiological state
Across all three subjects and all four facial regions, T1 showed dramatically lower signal amplitude (RMS) than T2 and T3. The gap was not subtle — T2 and T3 were 3–8× louder than T1 in terms of blood flow variability. This pattern held without exception across every ROI and every subject tested.

### 2. T2 and T3 share a physiological signature that T1 does not
The dominant frequency of the blood flow signal shifted between T1 and T2/T3. In s1 and s2, T1 showed a dominant frequency of ~0.96–1.1 Hz, while T2 and T3 both converged to ~0.82 Hz. This suggests T2 and T3 are physiologically more similar to each other than either is to T1 — confirmed by cross-correlation analysis, where T2 vs T3 correlations were consistently more positive than either vs T1.

### 3. The pattern replicates across subjects
The most important finding is replicability. All three subjects showed the same ordering — T1 lowest, T2 and T3 higher — independently. This rules out the pattern being a single-subject artefact and suggests it reflects something real about the tasks themselves.

### 4. Waveform shapes differ between trials, not just amplitudes
Cross-correlations between trial pairs were universally low (|r| < 0.30), meaning the blood flow waveforms are not just scaled versions of each other — they are genuinely different in shape. This is important: it implies the task influences the *dynamics* of blood flow, not just the overall level.

---

## What This Suggests
The consistency of the T1 < T2/T3 pattern across subjects and ROIs strongly suggests that facial blood flow is sensitive to the type of task being performed. T1 appears to represent a lower-demand or lower-arousal condition relative to T2 and T3. Whether the driver is cognitive load, emotional arousal, physical activity, or stress cannot be determined without knowing the task labels — but the physiological separation is clear.

---

## What We Cannot Conclude Yet
- **Causation:** We can see that blood flow differs between tasks, but we cannot say the task *caused* the difference without ruling out confounds (lighting changes, head movement, session order effects)
- **Mechanism:** Whether the signal reflects heart rate changes, vasomotor tone, emotional arousal, or motion artefacts is unresolved — BVP and EDA data exist and should be cross-referenced
- **Generalisability:** 3 subjects is a pilot. The pattern is promising but needs replication in s4–s10 before drawing broader conclusions
- **Classification:** The data is not yet sufficient to train a reliable classifier — overfitting to 3 individuals is near-certain at this sample size

---

## Recommended Next Steps

| Priority | Step | Why |
|---|---|---|
| 1 | Cross-reference BVP and EDA data | Validate that the facial blood flow signal reflects real physiology, not artefact |
| 2 | Identify task labels (T1, T2, T3) | Required to interpret the physiological meaning of the findings |
| 3 | Run s4–s10 | Build enough data for statistical inference and eventual classification |
| 4 | Pilot classifier (SVM/LDA) | Once n ≥ 8–10 subjects, test whether a simple model can classify task from blood flow features |
| 5 | Add head-motion correction | Improve signal quality by compensating for head movement |
