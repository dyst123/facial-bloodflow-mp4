# Facial Blood Flow from Video

Extracts and plots **relative facial blood flow signals** from an ordinary MP4 or AVI video — no special hardware required.

从普通 MP4 或 AVI 人脸视频中提取并绘制**相对面部血流信号**，无需特殊硬件。

---

## How to use / 使用方法

1. 把视频文件拖入 `Data/` 文件夹
2. 在终端运行：
```bash
cd src
python run_bloodflow.py
```
3. 结果图片自动保存在 `outputs/<视频名>/` 文件夹里

支持格式：`.mp4`、`.avi`、`.mov`、`.mkv`

---

## What it does / 功能说明

### English

Given a face video, the script:

1. Detects 478 facial landmarks per frame using **MediaPipe FaceMesh**
2. Extracts four regions of interest (ROIs): nose, forehead, left cheek, right cheek
3. Records the **average green channel intensity** inside each ROI for every frame
4. Converts those intensity time-series into **relative blood flow signals** using the Beer-Lambert relationship: `bf = -log(I)`
5. Smooths the signal with a moving average and zero-means it
6. Saves plots to `outputs/<video_name>/`

The green channel is used because blood (haemoglobin) absorbs green light most strongly — when blood flows through the skin, green reflectance drops slightly, producing a measurable signal. This technique is called **rPPG (remote photoplethysmography)**.

### 中文

给定一段人脸视频，脚本执行以下步骤：

1. **检测面部关键点** — 使用 MediaPipe FaceMesh，每帧检测 478 个面部坐标点
2. **划定 ROI 区域** — 根据关键点圈出四个皮肤区域：鼻子、额头、左脸颊、右脸颊
3. **提取绿色通道亮度** — 每帧记录每个区域内像素的平均绿色通道值
4. **转换为血流信号** — 使用 Beer-Lambert 公式 `bf = -log(绿色亮度)` 将亮度变化转化为血流变化信号
5. **平滑处理** — 移动平均去噪，减去均值得到相对变化量
6. **保存图表** — 输出各区域血流波形图

**为什么用绿色通道？** 血红蛋白对绿光的吸收最强。当皮肤下血流量增加时，皮肤对绿光的反射减少，亮度轻微下降。通过记录这个微小变化，就能还原血流的时间序列信号。这项技术叫做 **rPPG（远程光电容积脉搏波描记法）**。

---

## Output / 输出文件

Three plots are saved per video / 每个视频生成三张图：

| 文件名 | 内容 |
|--------|------|
| `bloodflow_all_rois_<name>.png` | 四个区域（鼻子/额头/左脸颊/右脸颊）各自的血流波形 |
| `bloodflow_nose_forehead_<name>.png` | 鼻子减去额头的差值信号（消除全局光照影响） |
| `bloodflow_forehead_<name>.png` | 仅额头血流（与原始论文图表格式一致） |

Y 轴为**相对血流（任意单位）**，反映血流浓度随时间的变化趋势，不是绝对流量值。

---

## How the blood flow signal is computed / 信号计算原理

```
每帧 ROI 区域的绿色通道平均亮度 I
        ↓
  bf = -log(I)          ← Beer-Lambert：血多 → 吸收绿光多 → 亮度低 → bf 升高
        ↓
  移动平均平滑           ← 去除高频噪声
        ↓
  减去均值               ← 变成相对变化量（零均值）
        ↓
  相对血流信号
```

不同区域做差（如鼻子 − 额头）可以消除全局光照变化对信号的干扰，突出局部血流变化。

---

## Where this came from / 代码来源

This code is adapted from the **[MNI-LAB/facial-bloodflow-tof](https://github.com/MNI-LAB/facial-bloodflow-tof)** repository, specifically the `Android-Testing` branch (`android/facial-bloodflow-mp4-processing/`).

本代码改编自 **[MNI-LAB/facial-bloodflow-tof](https://github.com/MNI-LAB/facial-bloodflow-tof)** 仓库的 `Android-Testing` 分支（`android/facial-bloodflow-mp4-processing/` 目录）。

原始代码基于 **Chronoptics KEA 飞行时间（ToF）相机**开发，该相机可同时捕获红外强度和深度数据。完整流程使用深度信息对头部运动进行补偿，从而得到更精确的血流估计。

本仓库移除了对 ToF 相机的依赖，改为处理**普通 RGB 视频**，牺牲了深度补偿换取更广泛的适用性。核心信号处理逻辑（ROI 提取、Beer-Lambert 转换、平滑处理、绘图）来自原始代码。

原始研究背景是**驾驶员监测**：通过面部血流信号检测疲劳、压力和生理状态。

---

## Limitations / 局限性

- **无深度补偿** — 原始版本用 ToF 深度数据修正头部运动，本版本没有，头动会影响信号质量
- **信号质量依赖光照** — 需要稳定、均匀的光照环境
- **输出为相对值** — 不是校准过的物理单位（ml/min 等）
- **需要正脸朝向镜头** — 侧脸或遮挡会导致检测失败
