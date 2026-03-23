import os
import sys
import time
import cv2
import numpy as np
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newp3_RGB_1 import NewPhase3
from p1and2_mp4_rgbonly import FaceProcessingRGBOnly

FPS = 30
CLIP_DUR = 20  # seconds

DATA_DIR = "./Data"
OUTPUT_DIR = "./outputs"
CSV_PATH = "./heart_rate_predictions_rgbonly.csv"

image_width = 640
image_height = 480

def read_video_frames(file_path, width, height):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def main():
    start_time = time.time()
    total_files = 0
    frames_per_interval = CLIP_DUR * FPS

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_name', 'interval', 'heart_rate'])

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        rgb_files = [f for f in filenames if 'rgb' in f and f.endswith('.mp4')]
        for rgb_file in rgb_files:
            base_name = rgb_file.replace('.mp4', '')
            rgb_path = os.path.join(dirpath, rgb_file)
            save_path = os.path.join(OUTPUT_DIR, base_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print(f'Processing {rgb_file}')
            start_time_file = time.time()

            rgb_frames = read_video_frames(rgb_path, image_width, image_height)
            total_frames = len(rgb_frames)
            print(f"Loaded {total_frames} RGB frames.")

            num_intervals = total_frames // frames_per_interval
            if total_frames - num_intervals * frames_per_interval > 10 * FPS:
                num_intervals += 1

            for interval in range(num_intervals):
                start_frame = interval * frames_per_interval
                end_frame = min((interval + 1) * frames_per_interval, total_frames)
                interval_rgb_frames = rgb_frames[start_frame:end_frame]

                face_proc = FaceProcessingRGBOnly(
                    interval_rgb_frames,
                    interval_start=0,
                    interval_end=len(interval_rgb_frames),
                    intervalMode=True,
                    visualize_FaceMesh=False,
                    visualize_ROIs=True,
                    image_width=image_width,
                    image_height=image_height,
                    fps=FPS,
                    getRR=True
                )

                I, D, RGB, chest_depths, all_ears, fbf_depths, fbf_ints, frame_count = face_proc.process_batches(
                    batch_size=16,
                    visualize_ROI=True,
                    visualize_FaceMesh=False
                )

                #print("I shape:", np.shape(I))
                #print("RGB shape:", np.shape(RGB))
                #print("frame_count:", frame_count)



                I = np.atleast_1d(I)
                D = np.zeros_like(I)
                RGB = np.atleast_2d(RGB)
                chest_depths = np.atleast_1d(chest_depths)
                all_ears = np.atleast_1d(all_ears)
                fbf_depths = np.atleast_2d(fbf_depths)
                fbf_ints = np.atleast_2d(fbf_ints)
                # Create dummy depth data as zeros
                D = np.zeros_like(I)

                #print("I shape:", np.shape(I))
                #print("RGB shape:", np.shape(RGB))
                #print("frame_count:", frame_count)

                phase3 = NewPhase3(
                    I, D, RGB, fbf_depths, fbf_ints, all_ears, chest_depths,
                    getRR=True, fps=FPS, frame_num=frame_count, 
                    timeWindow=CLIP_DUR, Window=True,
                    clip=f"{base_name}_interval_{interval+1}",
                    output_DIR=save_path
                )

                heart_rate = phase3.run(plot_fft=False)
                print(f"Interval {interval+1}/{num_intervals} Heart Rate: {heart_rate:.2f} bpm")

                with open(CSV_PATH, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([base_name, interval+1, heart_rate])

            end_time_file = time.time()
            print(f'Time taken to process file: {np.round((end_time_file - start_time_file), 2)} seconds')
            total_files += 1

    end_time = time.time()
    print('---------------------------------------------')
    print('-------------All files processed.------------')
    print('---------------------------------------------')
    print(f'Total time taken: {np.round((end_time - start_time), 2)} seconds')
    print(f'Total number of files processed: {total_files}')
    if total_files > 0:
        print(f'Average time taken per file: {np.round(((end_time - start_time) / total_files), 2)} seconds')

if __name__ == '__main__':
    main()