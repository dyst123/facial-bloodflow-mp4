import os
import cv2
import csv
import time
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.spatial import distance as dist
from scipy.io import savemat, loadmat
from PIL import Image, ImageDraw
from typing import ChainMap, Tuple, Union


class NewPhase3():
    """
    This class takes in the output (average intensity and depth values) from phase 2, the number of frames,
    a time window size and window boolean. The number of frames and time window are used in the mathematical
    operations involved in the Heart Rate calculation. The window boolean determines if a Hanning operation
    needs to be applied to the compensated intensity.
    
    The final output of this class is the Heart Rate.
    """

    def __init__(self, I, D, fbf_depths, fbf_intensities, all_ear_values, chest_depths, getRR = True, fps = 10, frame_num = 300, timeWindow = 30, Window = True, clip = '', output_DIR = ''):
        
        self.I = I[0]
        self.D = D[0]
        self.fbf_depths = fbf_depths
        self.fbf_intensities = fbf_intensities
        self.all_ear_values = all_ear_values[0]
        self.chest_depths = chest_depths
        self.getRR = getRR
        self.fps = fps
        self.frame_num = frame_num
        self.timeWindow = timeWindow
        self.Window = Window
        self.clip = clip
        self.output_DIR = output_DIR
    
    def run(self, plot_fft=False):
        """
        This function runs the phase 3 calculations. It calculates the Heart Rate using the FFT method and 
        the STFT method. It also calculates the Respiratory Rate, number of breaths, number of blinks, and PERCLOS.

        Returns
        -------
        HR_2D:
            Heart Rate calculated without depth compensation.
        HR: 
            Heart Rate calculated using the FFT method.
        Hr_stft: 
            Heart Rate calculated using the STFT method.
        hr_welch: 
            Heart Rate calculated using the Welch method.
        RR: 
            Respiratory Rate in breaths per minute scalar value.
        breathCount: 
            Number of breaths counted in the clip.
        blinks: 
            Number of blinks in the clip.
        perclos: 
            Percentage of eye closure in the clip.
        bf:
            Facial Blood Flow (Nose - Forehead).
        bf2:
            Facial Blood Flow (Cheek - Forehead).
        """

        # Creating an empty list to store the various HR values calculated
        HRs = []   

        # Removing any unnecessary zeros in the average intensity and depth arrays
        I_raw_rec = np.squeeze(self.I)
        Depth_rec = np.squeeze(self.D)

        # Smoothing the signals
        Depth_rec = scipy.signal.savgol_filter(Depth_rec, 9, 2, mode='nearest')
        I_raw_rec = scipy.signal.savgol_filter(I_raw_rec, 5, 2, mode='nearest')



        ######## HR THROUGH STFT METHOD START ########
        # I_raw_rec1 = scipy.signal.savgol_filter(I_raw_rec, 5, 2, mode='nearest')
        # I_raw_rec1 = self.Depth_compensation(I_raw_rec1[:], Depth_rec[:], 2, self.fps)
        I_avg = self.get_I_avg(self.frame_num,self.fps,Depth_rec,I_raw_rec)
        Hr_stft = self.run_stft(Depth_rec,self.fps,I_avg)
        ######## HR THROUGH STFT METHOD END ########

        # Getting the depth-compensated intensity 
        I_comp_rec = self.Depth_compensation(I_raw_rec[:], Depth_rec[:], 2, self.fps)

        # If the Window boolean is set to true, a Hanning operation will be performed on the 
        # compensated intensity array.
        if self.Window:
            min_val = np.min(I_comp_rec)
            max_val = np.max(I_comp_rec)
            I_comp_rec = (I_comp_rec - min_val) / (max_val - min_val)

            window = np.hanning(len(I_comp_rec))
            I_comp_rec = I_comp_rec * window
        

        # FFT (Fourier transform calculation)
        Intensity_freq = np.fft.fft(I_comp_rec[:])

        X_final = np.abs(Intensity_freq)
        freq = np.fft.fftfreq(len(X_final), 1.0 / self.fps)*60.0
        print(freq)
        X_final[np.where(freq<40)]=0
        X_final[np.where(freq>190)]=0

        # HR calculation
        HR = ((np.argmax(X_final[math.floor(0.6667 * len(X_final) / self.fps):math.ceil(2.5 * len(X_final) / self.fps)]) + math.floor(
            0.6667 * len(X_final) / self.fps))* self.fps * 60.0 / len(X_final))
        

        # 2D heart rate
        # FFT (Fourier transform calculation)
        Intensity_freq2 = np.fft.fft(I_raw_rec[:])
        X_final2 = np.abs(Intensity_freq2)
        freq2 = np.fft.fftfreq(len(X_final2), 1.0 / self.fps)*60.0
        print("freq2: ", freq2)
        X_final2[np.where(freq2<40)]=0
        X_final2[np.where(freq2>190)]=0

        # HR calculation
        HR_2D = ((np.argmax(X_final2[math.floor(0.6667 * len(X_final2) / self.fps):math.ceil(2.5 * len(X_final2) / self.fps)]) + math.floor(
            0.6667 * len(X_final2) / self.fps))* self.fps * 60.0 / len(X_final2))
        
        hr_welch, freq_filtered_comp, hr_arr_comp = self.run_welch(I_raw_rec, self.fps)

        
        # process other info
        RR, breathCount, blinks, perclos, bf, bf2 = self.processRest()


        # Plot FFT Spectrum
        if plot_fft:
            plt.figure()
            X_final[np.where(freq < 40)] = 0
            X_final[np.where(freq > 150)] = 0
            plt.plot(freq, X_final)
            plt.xlim(40, 150)
            HRs.append(HR)
            plt.show()

        return HR_2D, HR, Hr_stft, hr_welch, RR, breathCount, blinks, perclos, bf, bf2
    
    def processRest(self, plot_EAR=False):
        """
        This function processes the rest of the data for the clip.
        It calculates the respiratory rate, eye tracking metrics, and facial blood flow.
        It also plots the data and saves the plots in the Data folder.

        Returns
        -------
        RR: 
            Respiratory rate in breaths per minute scalar value.
        breathCount: 
            Number of breaths counted in the clip.
        blinks: 
            Number of blinks in the clip.
        perclos: 
            Percentage of eye closure in the clip.
        bf:
            Facial Blood Flow (Nose - Forehead).
        bf2:
            Facial Bloof Flow (Cheek - Forehead).
        """
        RR = 0
        breathCount = 0
        if self.getRR:
            RR, breathCount = self.getRespitoryRate(self.chest_depths, Savgof=True, Lowpass=False, Window=False, realFFT = True)
            # print(f"Respiratory Rate: {self.RR}")
            # print(f"Breaths in Clip: {self.breathCount}")

        # EYE TRACKING CALCULATIONS
        # max_ear = max(self.all_ear_values) if len(self.all_ear_values)>0 else 0 #!Previous implementation
        max_ear = np.percentile(self.all_ear_values, 75) if len(self.all_ear_values)>0 else 0
        perclos_threshold = max_ear * 0.65
        
        # print(max_ear)
        # print(perclos_threshold)
        # print(all_ear_values)
        blink_threshold = max_ear * 0.6
        # print(blink_threshold)
        blink_state = 0
        previous_state = 0
        blinks = 0
        
        # Apply Moving Average Filter to the EAR values
        moving_window_size = 3
        self.all_ear_values = np.convolve(self.all_ear_values, np.ones(moving_window_size)/moving_window_size, mode='valid')
        
        # Visualize EAR values and thresholds
        if plot_EAR:
            plt.figure()
            plt.plot(self.all_ear_values)
            plt.axhline(y=perclos_threshold, color='r', linestyle='--', label='PERCLOS Threshold')
            plt.axhline(y=blink_threshold, color='g', linestyle='--', label='Blink Threshold')
            plt.legend()
            plt.title('EAR Values')
            plt.xlabel('Frame')
            plt.ylabel('EAR Value')
            plt.show()
        

        for ear in self.all_ear_values:
            if ear < blink_threshold:
                blink_state = 1  
            else:
                blink_state = 0  
            
            if blink_state == 0 and previous_state == 1:
                blinks += 1
            
            previous_state = blink_state

        perclos = sum(ear < perclos_threshold for ear in self.all_ear_values) / self.frame_num if self.frame_num > 0 else 0

        # print(f"Blinks: {blinks}, PERCLOS: {perclos * 100:.2f}%")

        ### Facial Blood Flow Calculation ###

        """
        APPLY DEPTH COMP HERE. DONT USE LINEAR MODEL. CHECK MATLAB CODE TO VERIFY IF INTENSITY NEEDS TO BE SMOOTHED.
        
        """

        T = 1/self.fps

        ########## Previous model ######
        linearModel = [-0.766825576784889, 591.153575311779]

        # Graph

        plt.close('all')
        # Figure 1: Forehead Raw Intensity vs Forehead Compensated Intensity
        plt.figure(2)
        plt.plot(self.fbf_intensities[0,:], label=' Intensity', color='r')
        plt.title(f'Nose BEFORE Intensity with Linear Model')
        plt.xlabel('Sample')
        plt.ylabel('Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_DIR, f'nose_before_cints_{self.clip}.png'), bbox_inches='tight')

        # Left and right cheek averaged
        lrc_i = (self.fbf_intensities[2] + self.fbf_intensities[3])/2
        lrc_d = (self.fbf_depths[2] + self.fbf_depths[3])/2

        before_icomp = np.array(self.fbf_intensities)
        before_d = np.array(self.fbf_depths)

        # Send two arrays into a csv file
        # Specify the file name
        fname = 'abcd.csv'

        i_nose = before_icomp[0,:].tolist()
        i_fore = before_icomp[1,:].tolist()
        d_nose = before_d[0,:].tolist()
        d_fore= before_d[1,:].tolist()

        # Write the array to the CSV file
        with open(fname, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Nose intensities"])
            writer.writerow(i_nose)
            writer.writerow(["Forehead intensities"])
            writer.writerow(i_fore)
            writer.writerow(["Nose depths"])
            writer.writerow(d_nose)
            writer.writerow(["Forehead depths"])
            writer.writerow(d_fore)


        # """New implementation attempt"""
        # self.fbf_intensities[1, :] = np.squeeze(self.fbf_intensities[1, :])
        # self.fbf_intensities[0, :] = np.squeeze(self.fbf_intensities[0, :])
        # self.fbf_intensities[2, :] = np.squeeze(self.fbf_intensities[2, :])
        # self.fbf_intensities[3, :] = np.squeeze(self.fbf_intensities[3, :])

        # self.fbf_depths[1, :] = np.squeeze(self.fbf_depths[1, :])
        # self.fbf_depths[0, :] = np.squeeze(self.fbf_depths[0, :])
        # self.fbf_depths[2, :] = np.squeeze(self.fbf_depths[2, :])
        # self.fbf_depths[3, :] = np.squeeze(self.fbf_depths[3, :])

        # lrc_i = np.squeeze(lrc_i)
        # lrc_d = np.squeeze(lrc_d)

        # # Smooth signals before
        # # depth
        # self.fbf_intensities[0,:] = scipy.signal.savgol_filter(self.fbf_intensities[0,:], 9, 2, mode='nearest')
        # self.fbf_intensities[1,:] = scipy.signal.savgol_filter(self.fbf_intensities[1,:], 9, 2, mode='nearest')
        # self.fbf_intensities[2,:] = scipy.signal.savgol_filter(self.fbf_intensities[2,:], 9, 2, mode='nearest')
        # self.fbf_intensities[3,:] = scipy.signal.savgol_filter(self.fbf_intensities[3,:], 9, 2, mode='nearest')
        # lrc_i = scipy.signal.savgol_filter(lrc_i, 9, 2, mode='nearest')

        # self.fbf_depths[0,:] = scipy.signal.savgol_filter(self.fbf_depths[0,:], 9, 2, mode='nearest')
        # self.fbf_depths[1,:] = scipy.signal.savgol_filter(self.fbf_depths[1,:], 9, 2, mode='nearest')
        # self.fbf_depths[2,:] = scipy.signal.savgol_filter(self.fbf_depths[2,:], 9, 2, mode='nearest')
        # self.fbf_depths[3,:] = scipy.signal.savgol_filter(self.fbf_depths[3,:], 9, 2, mode='nearest')
        # lrc_i = scipy.signal.savgol_filter(lrc_d, 9, 2, mode='nearest')
        # # intensity

        # min_nose_int = min(self.fbf_intensities[0, :])
        # min_fh_int = min(self.fbf_intensities[1, :])
        # min_lc_int = min(self.fbf_intensities[2, :])
        # min_rc_int = min(self.fbf_intensities[3, :])
        # min_lrc_int = min(lrc_i)
        
        # self.fbf_intensities[1,:] = self.Depth_compensation(self.fbf_intensities[1,:], self.fbf_depths[1,:], 2, self.fps)
        # self.fbf_intensities[0,:] = self.Depth_compensation(self.fbf_intensities[0,:], self.fbf_depths[0,:], 2, self.fps)
        # self.fbf_intensities[2,:] = self.Depth_compensation(self.fbf_intensities[2,:], self.fbf_depths[2,:], 2, self.fps)
        # self.fbf_intensities[3,:] = self.Depth_compensation(self.fbf_intensities[3,:], self.fbf_depths[3,:], 2, self.fps)
        # lrc_i= self.Depth_compensation(lrc_i, lrc_d, 2, self.fps)

        # self.fbf_intensities[1,:] = self.fbf_intensities[1,:] + np.min(self.fbf_intensities[1,:])*-1 + 1 + min_fh_int
        # self.fbf_intensities[0,:] = self.fbf_intensities[0,:] + np.min(self.fbf_intensities[0,:])*-1 + 1 + min_nose_int
        # self.fbf_intensities[2,:] = self.fbf_intensities[2,:] + np.min(self.fbf_intensities[2,:])*-1 + 1 + min_lc_int
        # self.fbf_intensities[3,:] = self.fbf_intensities[3,:] + np.min(self.fbf_intensities[3,:])*-1 + 1 + min_rc_int
        # lrc_i = lrc_i + np.min(lrc_i)*-1 + 1 + min_lrc_int
        

        # print()
        # print()
        # print("fbf forehead intensities: ", bc_forehead_b)
        """END OF New implementation attempt"""

        # ########## Previous model ######
        # I_comp = self.fbf_intensities / (self.fbf_depths * linearModel[0] + linearModel[1])
        # I_comp_norm = I_comp / np.mean(I_comp, axis=1, keepdims=True)
         

        # ########## Previous model ######
        # I_comp_lrc = lrc_i / (lrc_d * linearModel[0] + linearModel[1])
        # I_comp_norm_lrc = I_comp_lrc / np.mean(I_comp_lrc, axis=0, keepdims=True)
        # ########## Previous model ######

        # ROI indices:
        # 0: nose
        # 1: forehead
        # ########## Previous model ######
        # bc_forehead_b = self.smooth_facial(-np.log(I_comp_norm[1,:]))
        # bc_nose_b = self.smooth_facial(-np.log(I_comp_norm[0,:]))
        # bc_leftcheek_b = self.smooth_facial(-np.log(I_comp_norm[2,:]))
        # bc_rightcheek_b = self.smooth_facial(-np.log(I_comp_norm[3,:]))
        # bc_lrc_b = self.smooth_facial(-np.log(I_comp_norm_lrc))


        # """"New implementation attempt"""
        ######### DC model ######
        bc_forehead_b = self.smooth_facial(-np.log(self.fbf_intensities[1,:]))
        bc_nose_b = self.smooth_facial(-np.log(self.fbf_intensities[0,:]))
        bc_leftcheek_b = self.smooth_facial(-np.log(self.fbf_intensities[2,:]))
        bc_rightcheek_b = self.smooth_facial(-np.log(self.fbf_intensities[3,:]))
        bc_lrc_b = self.smooth_facial(-np.log(lrc_i))

        # Nose - Forehead Blooflow
        bf = bc_nose_b - bc_forehead_b
        # Cheek - Forehead Bloodflow
        bf2 = bc_nose_b - bc_lrc_b

        tb = np.arange(self.fbf_intensities.shape[1]) * T

        participant = self.clip.strip(".csf")

        plt.close('all')
        # Figure 1: Forehead Raw Intensity vs Forehead Compensated Intensity
        plt.figure(2)
        plt.subplot(2, 1, 1)  # First plot
        plt.plot(tb, before_icomp[0, :], label='Raw Intensity')
        plt.title(f'Nose Raw Intensity {participant}')
        plt.xlabel('Sample')
        plt.ylabel('Intensity')


        plt.subplot(2, 1, 2)  # Second plot
        plt.plot(tb, self.fbf_intensities[0,:], label='Compensated Intensity', color='r')
        plt.title(f'Nose Compensated Intensity with Linear Model {participant}')
        plt.xlabel('Sample')
        plt.ylabel('Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_DIR, f'nose_cints_{participant}.png'), bbox_inches='tight')

        #plt.savefig(f'forehead_george_emo.png', bbox_inches='tight') ### HARDCODED CHANGE LATER

        # plt.show()

        plt.close('all')
        # Figure 2: Forehead Bloodflow
        plt.figure(3)
        plt.plot(tb, bc_forehead_b)
        plt.title(f'Forehead Bloodflow {participant}')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Blood Flow (a.u.)')
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'forehead_{participant}.png'), bbox_inches='tight')

        # plt.show()
        plt.close('all')
        # Figure 3: Nose Bloodflow
        plt.figure(4)
        plt.plot(tb, bc_nose_b)
        plt.title(f'Nose Bloodflow {participant}')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Blood Flow (a.u.)')
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'nose_{participant}.png'), bbox_inches='tight')
        # plt.show()
        
        
        # Figure 4: Left Cheek Bloodflow
        plt.close('all')
        plt.figure(5)
        plt.plot(tb, bc_leftcheek_b)
        plt.title(f'Left Cheek Bloodflow {participant}')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Blood Flow (a.u.)')
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'leftcheek_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER
        # plt.show()
        
        
        # Figure 5: Right Cheek Bloodflow
        plt.close('all')
        plt.figure(6)
        plt.plot(tb, bc_rightcheek_b)
        plt.title(f'Right Cheek Bloodflow {participant}')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Blood Flow (a.u.)')
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'rightcheek_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER
        # plt.show()

        # Figure 6: AVG Left and Right Cheek Bloodflow
        plt.close('all')
        plt.figure(7)
        plt.plot(tb, bc_lrc_b)
        plt.title(f'Avg. Left and Right Cheek Bloodflow {participant}')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Blood Flow (a.u.)')
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'avgleftrightcheek_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER
        # plt.show()


        # Close all figures
        plt.close('all')

        # Figure 7: Nose-Forehead Blood Flow
        plt.figure()
        plt.plot(tb, bf, label='Nose-Forehead')
        plt.title(f'Nose-Forehead Bloodflow of {self.clip}')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Blood Flow (a.u.)')
        #plt.ylim(-0.07, 0.07)
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'nose_forehead_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER

        # Raw depths
        # Close all figures
        plt.close('all')
        plt.figure()
        plt.plot(tb, self.fbf_depths[0,:], label='Nose')
        plt.title(f'Nose Depths {self.clip}')
        plt.xlabel('Frame')
        plt.ylabel('Depth (mm)')
        #plt.ylim(-0.07, 0.07)
        plt.savefig(os.path.join(self.output_DIR, f'nose_depths_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER

        # Close all figures
        plt.close('all')
        plt.plot(tb, self.fbf_depths[1,:], label='Forehead')
        plt.title(f'Forehead Depths {self.clip}')
        plt.xlabel('Frame')
        plt.ylabel('Depth (mm)')
        #plt.ylim(-0.07, 0.07)
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'forehead_depths_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER

        plt.close('all')
        plt.close('all')
        plt.plot(tb, self.fbf_intensities[1,:], label='Forehead')
        plt.title(f'Forehead Intensities {self.clip}')
        plt.xlabel('Frame')
        plt.ylabel('intensity')
        #plt.ylim(-0.07, 0.07)
        plt.legend()
        plt.savefig(os.path.join(self.output_DIR, f'forehead_ints_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER

        # Close all figures
        plt.close('all')
        plt.figure()
        plt.plot(tb, self.fbf_intensities[0,:], label='Nose')
        plt.title(f'Nose Intensities {self.clip}')
        plt.xlabel('Frame')
        plt.ylabel('intensity')
        #plt.ylim(-0.07, 0.07)
        plt.savefig(os.path.join(self.output_DIR, f'nose_ints_{participant}.png'), bbox_inches='tight') ### HARDCODED CHANGE LATER

        self.fbf_intensities = np.array(before_icomp)

        # Save the plot
        #plt.savefig(f'{savepath}nose_forehead_bloodflow_{self.clip}.png', bbox_inches='tight')

        #plt.show() # TURN ON TO SHOW GRAPH

        return RR, breathCount, blinks, perclos*100, bf, bf2

    def getRespitoryRate(self, chestDepths, Savgof=True, Lowpass=False, Window=False, realFFT=False, plot=False):
        """
        Calculates respiratory rate from chest depth data

        Args:
            chestDepths: list of average chest depths
            filename: filename string of file being processed
            Savgof: boolean flag to apply Savgol filter
            Lowpass: boolean flag to apply lowpass filter
            Window: boolean flag to apply windowing
            realFFT: boolean flag to apply real FFT

        Returns:
            RR: respiratory rate in breaths per minute scalar value
        """
        num_frames = len(chestDepths)

        if plot:
            # Create a figure with 3 subplots to plot 
            fig, axes = plt.subplots(1, 4, figsize=(15, 6))
            fig.suptitle(f'Chest Depth Data {self.clip}')
            time = np.linspace(0, 30, len(chestDepths))

            # Plot the raw chest depth data
            axes[0].plot(time, chestDepths)
            axes[0].set_title("Chest Depth Data")
            axes[0].set_xlabel("Time (seconds)")
            axes[0].set_ylabel("Chest Depth (mm)")
            fig.show()

        if Lowpass:
            chestDepths = self.apply_lowpass_filter(chestDepths, 1, self.fps)

        if Savgof:
            chestDepths = scipy.signal.savgol_filter(
                chestDepths, self.fps * 2.5, 2, mode='nearest')  # Window size 2.5 seconds

            if plot:
                # Plot the Savgol filtered chest depth data
                axes[1].plot(time, chestDepths)
                axes[1].set_title("Chest Depth Savgol Filtered")
                axes[1].set_xlabel("Time (seconds)")
                axes[1].set_ylabel("Chest Depth")
        

        # Applying fourier transform to the chest depth data
        T = 1/self.fps
        if realFFT:
            yf_rr1 = abs(np.fft.rfft(chestDepths))
            # yf_rr1 = 2.0 / num_frames * yf_rr1[:num_frames // 2]
            # xf_rr1 = np.linspace(0.0, 1.0 / (2.0 * T), num_frames // 2)
            N = chestDepths.size
            xf_rr1 = np.fft.rfftfreq(N, d=T)
            xf_rr2 = xf_rr1
            xf_rr1 = xf_rr1 * 60 # Breaths per minute

            xf_rr2 = xf_rr2 * (self.frame_num/self.fps)  # Breaths counted in clip
            #print(xf_rr1)
        else:
            yf_rr1 = abs(np.fft.fft(chestDepths))
            N = chestDepths.size
            xf_rr1 = np.fft.fftfreq(N, d=T)
            xf_rr2 = xf_rr1
            xf_rr1 = xf_rr1 * 60 # Breaths per minute

            xf_rr2 = xf_rr2 * (self.frame_num/self.fps)  # Breaths counted in clip


            #yf_rr1 = 2.0 / num_frames * yf_rr1[:num_frames // 2]
            #xf_rr1 = np.linspace(0.0, 1.0 / (2.0 * T), num_frames // 2)      

        yf_rr2 = yf_rr1
        yf_rr1[np.where(xf_rr1 <= 6)] = 0
        yf_rr1[np.where(xf_rr1 >= 30)] = 0

        yf_rr2[np.where(xf_rr2 <= (6/(60/(self.frame_num/self.fps))))] = 0
        yf_rr2[np.where(xf_rr2 >= (30/(60/( self.frame_num/self.fps))))] = 0

        if plot:
            # Plot the Fourier Transform of the chest depth data
            axes[2].bar(xf_rr1, yf_rr1)
            axes[2].set_title("FFT of Chest Depth")
            axes[2].set_xlim((6,30))
            #axes[2].set_ylim((0,2))
            axes[2].set_xlabel("Frequency (Breaths per Minute)")
            axes[2].set_ylabel("Fourier Magnitude")

            axes[3].bar(xf_rr2, yf_rr2)
            axes[3].set_title("FFT of Chest Depth")
            axes[3].set_xlim(( 6/(60/(self.frame_num/self.fps))) , (30/(60/( self.frame_num/self.fps)) ))
            #axes[2].set_ylim((0,2))
            axes[3].set_xlabel("Frequency (Breaths In Clip)")
            axes[3].set_ylabel("Fourier Magnitude")

            fig.show()
            plt.tight_layout()
            plt.show()

        peaks, properties = scipy.signal.find_peaks(yf_rr1)
        max_index = np.argmax(yf_rr1[peaks])

        RR = xf_rr1[peaks[max_index]]

        peaks, properties = scipy.signal.find_peaks(yf_rr2)
        max_index = np.argmax(yf_rr2[peaks])
        
        BreathCountInClip = xf_rr2[peaks[max_index]]

        # with open('thanos_results.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([self.filename, RR])

        return RR, BreathCountInClip

    def apply_lowpass_filter(self, signal, cutoff_frequency, sampling_rate, filter_order=2):
        nyquist_frequency = 0.5 * sampling_rate
        normalized_cutoff = cutoff_frequency / nyquist_frequency
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def smooth_facial(self,x, window_len=20):
          """
          Calculates the final facial bloodflow
          
          Args:
              x (2D Array of ints) : compensated intensity
              
          Returns:
              Facial bloodflow signal
          """
          
          s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
          w = np.ones(window_len, 'd')
          y = np.convolve(w/w.sum(), s, mode='valid')
          return y[window_len//2:-window_len//2+1]
    
    def Depth_compensation(self, I_raw, Depth, timeWindow, Fs):
        """
        Depth_compensation finds depth-compensated intensity using the equation in the original research paper
        Link: https://www.researchgate.net/publication/367720861_Contactless_Vital_Sign_Monitoring_System_for_In-Vehicle_Driver_Monitoring_Using_a_Near-Infrared_Time-of-Flight_Camera

        Args:
            I_raw (2D Array of ints): Raw intensities at each ROI
            Depth (2D Array of ints): Raw depths at each ROI
            timeWindow (int): Every time window to iterate for finding best b value, in seconds
            Fs (int): frames per second

        Returns:
            I_comp (2D Array of ints): Compensated intesities (7x1800 for 60s)

        """
        I_comp = np.ones_like(I_raw)
        
        # best: scalar variable to find best b value
        best = 1
        
        # best_rem: scalar variable to find best b value for the remainder of the clip less than 20s
        best_rem = 1
        
         # Iterate through the different ROIs
        
        for ROI in range(1):
            # I_comp_ROI: 2D array of ints with the compensated intensities for the ROI
            I_comp_ROI = np.ones(I_raw.shape[0])
            # i: scalar variable to iterate through each clip(time window)
            i = 1
            
            # Iterate through every clip for a few frames as determined by the timeWindow value
            while (i * (timeWindow * Fs)) < len(I_raw[ :]):
                # cor: the lowest correlation coefficient that we compare to/reset (we start at 1 because that is highest possible value)
                cor = 2
                
                # For each clip, iterate through different b values with a = 1
                for bi in np.arange(0.2, 5.01, 0.01):
                    # for bi in np.arange(0.3, 4.1, 0.1):
                    bI_comp = I_raw[ ((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))] / (
                            (Depth[((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))]) ** (-bi))
                    # Find correlation between bI_comp and Depth
                    corr_v = np.corrcoef(bI_comp, Depth[((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))])
                    # Take absolute value of correlation coefficients
                    corr_ = abs(corr_v[1, 0])
                    
                    # If the new correlation coeff is less than the old one, reset cor value and best I_comp
                    if corr_ < cor:
                        cor = corr_
                        best = bI_comp
                
                # Normalize data using z-scores
                I_comp_ROI[((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))] = (best - np.mean(best)) / np.std(best)
                i += 1
            
            # Repeat the above procedure for the remainder of the clip, if any
            cor = 2
            for bii in np.arange(0.2, 5.1, 0.1):
                # for bii in np.arange(0.3, 4.1, 0.1):
                bI_comp = I_raw[(((i - 1) * (timeWindow * Fs))):] / (Depth[ (((i - 1) * (timeWindow * Fs))):]) ** (
                -bii)
                # Find correlation between bI_comp and Depth
                corr_v = np.corrcoef(bI_comp, Depth[ (((i - 1) * (timeWindow * Fs))):])
                # Take absolute value of correlation coefficients
                corr_ = abs(corr_v[1, 0])
                
                # If the new correlation coeff is less than the old one, reset cor value and I_comp
                if corr_ < cor:
                    cor = corr_
                    best_rem = bI_comp
            
            # Normalize data
            I_comp_ROI[(((i - 1) * (timeWindow * Fs))):] = (best_rem - np.mean(best_rem)) / np.std(best_rem)
            # Append to final output matrix
            I_comp[:] = I_comp_ROI
            
        return I_comp
    
    def run_stft(self, D_signal_smooth, fps, I_avg, plot=False):
        """
        This calculates the short-time Fourier transform (STFT) over the given average 
        intensity using the in-built short-time fourier transform function in Python.

        Args: 
           D_signal_smooth (2D Array of ints) : The smoothened depth signal. Not being used for the STFT calculation. Currently this is being used in
           the experimental motion-scoring code
           I_avg (2D Array of ints) : The depth-compensated average intensity
           fps (int) : The frames per second for the clip

        Returns:
            HR_stft_final (int) : STFT attempt's HR
        """
        
        # The segment length and segment overlap values which are used in the STFT calculation to determine the Hop size as per the 
        # usual mathematical STFT equation.
        seg_length= 112 #126 (previous value for 10s interval at 15fps)
        seg_overlap= 110 #123 (previous value for 10s interval at 15fps)

        ######################## Motion scoring calculation attempt #################################
        mean_dist=np.average(D_signal_smooth[12:-12])
        Sig=np.zeros(D_signal_smooth[12:-12].shape[0]+seg_length)+mean_dist
        Sig[int(seg_length/2):int(seg_length/2)+D_signal_smooth[12:-12].shape[0]]=D_signal_smooth[12:-12]
        number_of_seg=int((Sig.shape[0]-seg_length)/(seg_length-seg_overlap))+1
        seg_time=np.zeros((seg_length,number_of_seg))
        motion_score_seg=np.zeros(number_of_seg)
        mean_postion_seg=np.average(seg_time,axis=1)
        ######################## Motion scoring calculation attempt end #################################

        # Calculating the STFT
        f, t_stft, Zxx = signal.stft(I_avg[12:-12], fs=fps, window='hann', nperseg=seg_length, noverlap=seg_overlap, padded=False)
        
        # Plotting, if required
        # plt.pcolormesh(t_stft, f*60, np.abs(Zxx), vmin=0, vmax=0.3, shading='gouraud')
        # plt.ylim(30,200)
        # plt.imshow(abs(coef),extent=[1/30, 577/30, 1, 65 ], cmap='jet', aspect='auto', vmax=(abs(coef)).max(), vmin=0) # doctest: +SKIP
        # plt.show()
        
        # HR calculation using the STFT-operated, depth-compensated intensity
        HR_stft_max=np.zeros(Zxx.shape[1])
        HR_stft_2max=np.zeros(Zxx.shape[1])
        max_f_stft_max=np.argsort(np.abs(Zxx),axis=0)
        HR_stft_max=f[max_f_stft_max[-1,:]]*60
        HR_stft_2max=f[max_f_stft_max[-2,:]]*60
        HR_stft=np.hstack((HR_stft_max,HR_stft_2max))
        HR_stft_final = np.median(HR_stft_max)

        # Plotting STFT peaks
        if plot:
            plt.figure(3)
            plt.plot(t_stft, HR_stft_max, label='HR stft max')
            plt.title('STFT for ' + self.clip)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.show()
        
        return HR_stft_final
    
    def run_welch(self, intensity, fps, minHz=0.65, maxHz=4.0, nfft=2048):
        """
        This function computes Welch method for spectral density estimation.

        Parameters
        ----------
        intensity : flaot32 numpy.ndarray 
            The intensity frame of the clip.
        fps : float
            Frames per seconds of the clip.
        minHz : float
            Frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz : float
            Frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft : int
            Number of DFT points, specified as a positive integer.

        Returns
        --------
        hr : 
            
        freq_filtered_comp : 
        
        hr_arr_comp : 
            
        """
        n = len(intensity)
        if n < 256:
            seglength = n
            overlap = int(0.8*n)  # fixed overlapping
        else:
            seglength = 256
            overlap = 200
        # -- periodogram by Welch
        F, P = scipy.signal.welch(intensity, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
        F = F.astype(np.float32)
        P = P.astype(np.float32)
        # -- freq subband (0.65 Hz - 4.0 Hz)
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        freq_filtered_comp = 60*F[band]
        hr_arr_comp = P[band]
        # get max bpm
        hr = freq_filtered_comp[np.argmax(hr_arr_comp)]
        
        return hr, freq_filtered_comp, hr_arr_comp
    
    def get_I_avg(self, frame_num, fps, D_signal_smooth, I_signal_smooth):
        """
        This function calculates the average depth compensated intensity to be used for the STFT process.
        
        Args:
             frame_num (int) : Number of frames in clip
             fps (int) : Frames per second in the clip
             D_signal_smooth (2D Array of ints) : The smoothened depth signal array
             I_signal_smooth (2D Array of ints) : The smmothened intensity signal array
             
        Returns:
            I_avg (2D array of ints) : Depth-compensated avaerage intensity signal
        """
        
        # Since only one ROI is being used
        roi_region = 1
        
        frame_num_HR=frame_num

        # Since we are only using cheek_n_nose, it's just 1 roi region
        I_compensated=np.zeros((roi_region,frame_num_HR)) 
        I_2alg=np.zeros((roi_region,frame_num_HR))
        
        # Compensating for depth
        T = 1.0 / fps   
        D_signal_smooth[np.where(np.isnan(D_signal_smooth)>0)]=0
        I_signal_smooth[np.where(np.isnan(I_signal_smooth)>0)]=0
        yfd = abs(fft(D_signal_smooth-np.mean(D_signal_smooth)))
        yfd=2.0 / frame_num * yfd[:D_signal_smooth.shape[0] // 2]
        yfd=yfd-np.mean(yfd)
        xfd = np.linspace(0.0, 1.0 / (2.0 * T), D_signal_smooth.shape[0] // 2)
        xfd = xfd * 60
        filters=np.exp(-0.000010*(xfd-85)**2)
        yfd= yfd*filters   
        yfd[np.where(xfd<=40 )]=0
        yfd[np.where(xfd>=150)]=0
        peaks_d, properties_d = scipy.signal.find_peaks(yfd)
        max_index=np.argmax(yfd[peaks_d])
        tw=1/(xfd[peaks_d[max_index]]/30)
         
        I_compensated[0][12:-12] = self.distcomp(I_signal_smooth[12:-12]/120, D_signal_smooth[12:-12],time_window=1, Fs = fps)
        I_compensated[0, np.where(np.isnan(I_compensated[0])>0)]=0
        I_2alg[0]=I_compensated[0]
        I_2alg[0, np.where(np.isnan(I_2alg[0])>0)]=0

        I_avg=np.average(I_2alg,axis=0)
        
        I_comp_cont=np.zeros((1)) # stores intensity compensated values 

        I_comp_cont=np.hstack((I_comp_cont, I_compensated[0][12:-12]))
        sos = signal.butter(4, [30/60, 200/60], 'bandpass', analog=False, output='sos', fs=fps)
        I_avg = signal.sosfilt(sos, I_avg)

        return I_avg
    
    def distcomp (self, roimean1, distmean1, power_range=np.arange(0.3,4.1,0.1), lco_range=np.arange(0.2,5.025,0.025),time_window=1, Fs=10):
        """
        This function is used explicitly inside the get_I_avg function to calculate the depth 
        compensated intensity using the equation in the original paper. Since the STFT section 
        is still being experimented on, a separate depth compensation function was made for this.
        
        Args:
            roimean1 (2D array of ints) : The intensity signal that is to be compensated for depth
            distmean1 (2D array of ints) : The depth signal that is to be used in the calculation
            
        Returns:
            neutralized_pre (2D array of ints) : Depth compensated intensity
        """
        
        # The timewidow over which the iteration will be performed
        timewindow=int(time_window*Fs);  
        L=len(roimean1)
        num_window=math.floor(L/timewindow)
        
        neutralized_pre=np.zeros(len(roimean1))
        neutralized=np.zeros(len(roimean1))
        
        # Iterations start
        for i in range(num_window):
            neutralized_pre[i*timewindow:(i+1)*timewindow-1] = roimean1[i*timewindow:(i+1)*timewindow-1]*(distmean1[i*timewindow:(i+1)*timewindow-1]**0.5)
            correlation=np.corrcoef(neutralized_pre[i*timewindow:(i+1)*timewindow-1],distmean1[i*timewindow:(i+1)*timewindow-1])
            correlation_pre=abs(correlation[1,0])
            for ii in power_range:
                for iii in lco_range:
                    neutralized[i*timewindow:(i+1)*timewindow-1] = roimean1[i*timewindow:(i+1)*timewindow-1]*(iii*(distmean1[i*timewindow:(i+1)*timewindow-1]**ii))
                    correlation=np.corrcoef( neutralized[i*timewindow:(i+1)*timewindow-1],distmean1[i*timewindow:(i+1)*timewindow-1])
                    if abs(correlation[1,0])<correlation_pre:
                        neutralized_pre[i*timewindow:(i+1)*timewindow-1]=neutralized[i*timewindow:(i+1)*timewindow-1]
                        correlation_pre=abs(correlation[1,0]);
            neutralized_pre[i*timewindow:(i+1)*timewindow-1]=(neutralized_pre[i*timewindow:(i+1)*timewindow-1]-np.mean(neutralized_pre[i*timewindow:(i+1)*timewindow-1]))/np.std(neutralized_pre[i*timewindow:(i+1)*timewindow-1])
        if L%timewindow !=0:
            if L%timewindow >=2:
                neutralized_pre[int(L-L%timewindow):L] = roimean1[int(L-L%timewindow):L]*((distmean1[int(L-L%timewindow):L]**0.5))
                correlation=np.corrcoef(neutralized_pre[int(L-L%timewindow):L],distmean1[int(L-L%timewindow):L])
                correlation_pre=abs(correlation[1,0])
                for ii in power_range:
                    for iii in lco_range:
                        neutralized[int(L-L%timewindow):L] = roimean1[int(L-L%timewindow):L]*(iii*(distmean1[int(L-L%timewindow):L]**ii))
                        correlation=np.corrcoef( neutralized[int(L-L%timewindow):L],distmean1[int(L-L%timewindow):L])
                        if abs(correlation[1,0])<correlation_pre:
                            neutralized_pre[int(L-L%timewindow):L]=neutralized[int(L-L%timewindow):L]
                            correlation_pre=abs(correlation[1,0]);
            neutralized_pre[int(L-L%timewindow):L]=(neutralized_pre[int(L-L%timewindow):L]-np.mean(neutralized_pre[int(L-L%timewindow):L]))/np.std(neutralized_pre[int(L-L%timewindow):L])
        else:
            neutralized_pre[L-1]=neutralized_pre[L-2]
        return neutralized_pre


