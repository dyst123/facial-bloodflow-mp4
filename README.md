# Facial Blood Flow from Video

Extracts and plots **relative facial blood flow signals** from an ordinary MP4 or AVI video — no special hardware required.

---

## What it does

Given a face video, the script:

1. Detects 478 facial landmarks per frame using **MediaPipe FaceMesh**
2. Extracts four regions of interest (ROIs): nose, forehead, left cheek, right cheek
3. Records the **average green channel intensity** inside each ROI for every frame
4. Converts those intensity time-series into **relative blood flow signals** using the Beer-Lambert relationship: `bf = -log(I)`
5. Smooths the signal with a moving average and zero-means it
6. Saves plots to `outputs/<video_name>/`

The green channel is used because blood (haemoglobin) absorbs green light most strongly — when blood flows through the skin, green reflectance drops slightly, producing a measurable signal. This technique is called **rPPG (remote photoplethysmography)**.

---

## Output

Three plots are saved per video:

| File | Contents |
|------|----------|
| `bloodflow_all_rois_<name>.png` | Blood flow waveform for each of the 4 ROIs |
| `bloodflow_nose_forehead_<name>.png` | Nose minus forehead differential signal |
| `bloodflow_forehead_<name>.png` | Forehead blood flow only (matches published paper format) |

The Y-axis is **Relative Blood Flow (a.u.)** — a unitless signal that reflects changes in blood concentration over time, not absolute flow rate.

---

## How to run

**Install dependencies:**
```bash
pip install opencv-python mediapipe numpy matplotlib pillow
```

**Run on all videos in Data/:**
```bash
cd src
python run_bloodflow.py
```

**Run on a specific file:**
```bash
cd src
python run_bloodflow.py myvideo.mp4
python run_bloodflow.py recording.avi
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

Place input videos in the `Data/` folder. Results are saved to `outputs/<video_name>/`.

---

## How the blood flow signal is computed

```
Raw green intensity per frame (I)
        ↓
  bf = -log(I)          ← Beer-Lambert: more blood = less reflected light = higher bf
        ↓
  moving average smooth  ← removes high-frequency noise
        ↓
  subtract mean          ← zero-mean to get relative change
        ↓
  Relative Blood Flow signal
```

The difference between ROIs (e.g. Nose − Forehead) highlights localised blood flow changes, filtering out global illumination variation that affects all regions equally.

---

## Where this came from

This code is adapted from the **[MNI-LAB/facial-bloodflow-tof](https://github.com/MNI-LAB/facial-bloodflow-tof)** repository, specifically the `Android-Testing` branch (`android/facial-bloodflow-mp4-processing/`).

The original codebase was built around a **Chronoptics KEA Time-of-Flight (ToF) camera**, which captures both infrared intensity and depth data simultaneously. The full pipeline used depth information to compensate for subject movement (depth compensation), producing more accurate blood flow estimates.

This repo strips out the ToF camera dependency and adapts the pipeline to work with **ordinary RGB video**, sacrificing depth compensation in exchange for accessibility. The core signal processing logic — ROI extraction, Beer-Lambert conversion, smoothing, and plotting — is preserved from the original.

The original research context is driver monitoring: detecting fatigue, stress, and physiological state from facial blood flow in a vehicle setting.

---

## Limitations

- No depth compensation (original used ToF depth data to correct for head movement)
- Signal quality depends heavily on lighting consistency and minimal head motion
- Output is a relative signal only — not calibrated to physical units
- Requires a clear, well-lit frontal face view
