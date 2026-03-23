from unittest import result
import numpy as np
import scipy
import math
import os
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.spatial import distance as dist
from scipy.io import savemat, loadmat
from PIL import Image, ImageDraw
from typing import ChainMap, Tuple, Union

class NewPhase3():
    def __init__(self, I, D, RGB, fbf_depths, fbf_intensities, all_ear_values, chest_depths, getRR = True, fps = 10, frame_num = 300, timeWindow = 30, Window = True, clip = '', output_DIR = ''):
        self.I = I
        self.D = D
        self.RGB = RGB
        self.fbf_depths = fbf_depths
        self.fbf_intensities = fbf_intensities
        self.all_ear_values = all_ear_values
        self.chest_depths = chest_depths
        self.getRR = getRR
        self.fps = fps
        self.frame_num = frame_num
        self.timeWindow = timeWindow
        self.Window = Window
        self.clip = clip
        self.output_DIR = output_DIR

        # Plot RGB channels
        plt.figure()
        plt.plot(self.RGB[0], label='Red')
        plt.plot(self.RGB[1], label='Green')
        plt.plot(self.RGB[2], label='Blue')
        plt.title(f'RGB Channels {self.clip}')
        plt.xlabel('Frame')
        plt.ylabel('RGB Intensity')
        plt.legend()
        if self.output_DIR:
            plt.savefig(os.path.join(self.output_DIR, f'rgb_channels_{self.clip}.png'), bbox_inches='tight')
        plt.close('all')


    def safesavgol(self, signal, window_length=5, polyorder=2):
        signal = np.atleast_1d(signal)
        # Use a smaller window if the signal is short
        max_window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        window_length = min(window_length, max_window)
        if window_length < 3:
            window_length = 3
        if window_length <= polyorder:
            polyorder = window_length - 1
        if len(signal) < window_length:
            return signal
        return scipy.signal.savgol_filter(signal, window_length, polyorder, mode='nearest')
    
    def run(self, plot_fft=True):
        HRs = []
        I_raw_rec = np.atleast_1d(np.squeeze(self.I))
        Depth_rec = np.atleast_1d(np.squeeze(self.D))
        Green_raw_rec = np.atleast_1d(np.squeeze(self.RGB[1]))

        plt.figure()
        plt.plot(Green_raw_rec, label='Green Channel')
        plt.title("Green Channel Signal")
        plt.xlabel("Frame")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Smooth all signals safely
        I_raw_rec = self.safesavgol(I_raw_rec, window_length=11, polyorder=2)
        Depth_rec = self.safesavgol(Depth_rec, window_length=11, polyorder=2)
        Green_raw_rec = self.safesavgol(Green_raw_rec, window_length=20, polyorder=3)


        plt.figure()
        plt.plot(Green_raw_rec, label='Smoothed Green Channel')
        plt.title("Smoothed Green Channel Signal")
        plt.xlabel("Frame")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

     # Ensure all signals are the same length
        min_len = min(len(I_raw_rec), len(Depth_rec), len(Green_raw_rec))
        I_raw_rec = I_raw_rec[:min_len]
        Depth_rec = Depth_rec[:min_len]
        Green_raw_rec = Green_raw_rec[:min_len]
        I_comp_rec = self.Depth_compensation(I_raw_rec, Depth_rec, 2, self.fps)
        Green_comp_rec = self.Depth_compensation(Green_raw_rec, Depth_rec, 2, self.fps)


        # Get average intensities
        I_avg = self.get_I_avg(self.frame_num, self.fps, Depth_rec, I_raw_rec)
        Green_avg = self.get_I_avg(self.frame_num, self.fps, Depth_rec, Green_raw_rec)

        # Calculate HR using STFT
        Hr_stft = self.run_stft(Depth_rec, self.fps, I_avg)
        Hr_stft_green = self.run_stft(Depth_rec, self.fps, Green_avg)

        # Depth compensation
        I_comp_rec = self.Depth_compensation(I_raw_rec, Depth_rec, 2, self.fps)
        Green_comp_rec = self.Depth_compensation(Green_raw_rec, Depth_rec, 2, self.fps)

        #print("I_comp_rec:", I_comp_rec)
        #print("Green_comp_rec:", Green_comp_rec)
        print("Length of signal:", len(Green_comp_rec))
        print("Max value in Green_comp_rec:", np.max(Green_comp_rec))
        print("Min value in Green_comp_rec:", np.min(Green_comp_rec))
        if self.Window:
            min_val = np.min(I_comp_rec)
            max_val = np.max(I_comp_rec)
            min_val_green = np.min(Green_comp_rec)
     # FFT calculation
        Intensity_freq = np.fft.fft(I_comp_rec)
        Intensity_freq_green = np.fft.fft(Green_comp_rec)

        X_final = np.abs(Intensity_freq)
        X_final_green = np.abs(Intensity_freq_green)
        freq = np.fft.fftfreq(len(X_final), 1.0 / self.fps)*60.0
        freq_green = np.fft.fftfreq(len(X_final_green), 1.0 / self.fps)*60.0

        plt.figure()
        plt.plot(freq_green, X_final_green)
        plt.title("FFT of Green Channel")
        plt.xlabel("BPM")
        plt.ylabel("Amplitude")
        plt.xlim(40, 120)
        plt.grid(True)
        plt.show()

        # Filter frequencies
        X_final[np.where(freq<40)]=0
        X_final[np.where(freq>190)]=0
        X_final_green[np.where(freq_green<40)]=0
        X_final_green[np.where(freq_green>190)]=0
    # Calculate heart rates
        HR = ((np.argmax(X_final[math.floor(0.6667 * len(X_final) / self.fps):math.ceil(2.5 * len(X_final) / self.fps)]) + math.floor(
            0.6667 * len(X_final) / self.fps))* self.fps * 60.0 / len(X_final))
        HR_green = ((np.argmax(X_final_green[math.floor(0.6667 * len(X_final_green) / self.fps):math.ceil(2.5 * len(X_final_green) / self.fps)]) + math.floor(
            0.6667 * len(X_final_green) / self.fps))* self.fps * 60.0 / len(X_final_green))

        return HR_green  # Return green channel HR as it's typically most reliable

    def Depth_compensation(self, I_raw, Depth, timeWindow, Fs):
        I_raw = np.atleast_1d(I_raw)
        Depth = np.atleast_1d(Depth)
        signal_length = len(I_raw)
        if signal_length != len(Depth):
            raise ValueError("Signal lengths must match")
        
        power_range = np.arange(0.3, 4.1, 0.1)
        lco_range = np.arange(0.2, 5.025, 0.025)
        
        return self.distcomp(I_raw, Depth, power_range, lco_range, timeWindow, Fs)

    def get_I_avg(self, frame_num, fps, D_signal_smooth, I_signal_smooth):
        I_signal_smooth = np.atleast_1d(I_signal_smooth)
        polyorder = 2
        n = len(I_signal_smooth)
    # Calculate window size
        window_size = int(frame_num / (self.timeWindow * fps))
    # Ensure window_size is odd and greater than polyorder
        if window_size <= polyorder:
            window_size = polyorder + 1
        if window_size % 2 == 0:
            window_size += 1
        if window_size > n:
            window_size = n if n % 2 == 1 else max(3, n - 1)
        if window_size < 3:
            window_size = 3
    # If the signal is too short, just return the original signal
        if n < window_size:
            return I_signal_smooth
        return scipy.signal.savgol_filter(I_signal_smooth, window_size, polyorder)

    def run_stft(self, D_signal_smooth, fps, I_avg, plot=False):
        I_avg = np.atleast_1d(I_avg)
        n = len(I_avg)
        nperseg = min(fps * 4, n)
        noverlap = min(fps * 3, nperseg - 1) if nperseg > 1 else 0
        if nperseg < 2:
            return 0  # Not enough data for STFT
        f, t, Zxx = scipy.signal.stft(I_avg, fps, nperseg=nperseg, noverlap=noverlap)
        Zxx = np.abs(Zxx)
        f = f * 60

        mask = (f >= 40) & (f <= 190)
        f = f[mask]
        Zxx = Zxx[mask]
        if Zxx.shape[0] == 0:
            return 0
        hr_idx = np.argmax(np.mean(Zxx, axis=1))
        return f[hr_idx]
    
    def distcomp(self, roimean1, distmean1, power_range=np.arange(0.3,4.1,0.1), 
                 lco_range=np.arange(0.2,5.025,0.025), time_window=1, Fs=10):
        roimean1 = np.atleast_1d(roimean1)
        distmean1 = np.atleast_1d(distmean1)
        best_power = 1
        best_lco = 1
        # Always return an array, never a scalar
        result = roimean1 - best_lco * (distmean1 ** best_power)
        return np.atleast_1d(result)
    
    def draw_bounding_boxes(self, bounding_box_pixels, frame, roi_name):
    # bounding_box_pixels: (x1, y1, x2, y2)
        x1, y1, x2, y2 = bounding_box_pixels
        color = (0, 255, 0)  # Green box
        thickness = 2
        # Draw rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        # Optionally, add ROI name as text
        cv2.putText(frame, roi_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame