import cv2
import numpy as np
from facemeshmodule_new import FaceMeshDetector
from chestROIReverseEngineering import ChestROI
from scipy.signal import butter, filtfilt
from scipy.spatial import distance as dist

import numpy as np
from facemeshmodule_new import FaceMeshDetector
from chestROIReverseEngineering import ChestROI

class FaceProcessingRGBOnly:
    def __init__(
        self,
        frames,  # List or numpy array of RGB frames
        interval_start,
        interval_end,
        intervalMode=False,
        visualize_FaceMesh=False,
        visualize_ROIs=True,
        static_image_mode=False,
        image_width=804,
        image_height=672,
        fps=30,
        getRR=False
    ):
        self.frames = frames
        self.depth_frames = [np.zeros((image_height, image_width)) for _ in frames]  # Dummy depth frames
        self.intervalMode = intervalMode
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        self.visualize_FaceMesh = visualize_FaceMesh
        self.visualize_ROI = visualize_ROIs
        self.getRR = getRR
        self.static_image_mode = static_image_mode

        self.face_mesh_detector = FaceMeshDetector(static_image_mode=static_image_mode)
        self.chest_roi = ChestROI()

    def process_batches(self, batch_size=8, visualize_ROI=True, visualize_FaceMesh=False):
        intensity_avg = []
        depth_avg = []
        rgb_avg = [[], [], []]
        chest_depths = []
        all_ears = []
        fbf_depths = np.zeros((4, self.interval_end - self.interval_start))
        fbf_ints = np.zeros((4, self.interval_end - self.interval_start))
        frame_count = 0

        for idx in range(self.interval_start, self.interval_end):
            frame = self.frames[idx]
            depth_frame = self.depth_frames[idx]  # All zeros
            face_detected, landmarks = self.face_mesh_detector.find_face_mesh(frame, draw=self.visualize_FaceMesh)
            if landmarks is None or np.all(landmarks == 0):
                print(f"No face detected in frame")

                intensity_avg.append(0)
                depth_avg.append(0)
                rgb_avg[0].append(0)
                rgb_avg[1].append(0)
                rgb_avg[2].append(0)
                chest_depths.append(0)
                all_ears.append(0)
                fbf_depths[:, frame_count] = 0
                fbf_ints[:, frame_count] = 0
                frame_count += 1
                continue

            
            
            roi_points = self.get_bounding_box("cheek_n_nose", landmarks)  # shape (N, 2)
            
            bbox = self.get_face_bounding_box(roi_points)  # returns (x1, y1, x2, y2)
            
            frame_with_box = self.draw_bounding_boxes(bbox, frame.copy(), "cheek_n_nose")
            if visualize_ROI:
                
                cv2.imshow('Bounding Box', frame_with_box)
                cv2.waitKey(1)

            x1, y1, x2, y2 = bbox

            roi_pixels = frame[y1:y2, x1:x2].reshape(-1, 3)

            if roi_pixels is not None and len(roi_pixels) > 0:
                intensity_avg.append(np.mean(self.intensity_to_grayscale(roi_pixels)))
                rgb_avg[0].append(np.mean(roi_pixels[:, 0]))
                rgb_avg[1].append(np.mean(roi_pixels[:, 1]))
                rgb_avg[2].append(np.mean(roi_pixels[:, 2]))
            else:
                intensity_avg.append(0)
                rgb_avg[0].append(0)
                rgb_avg[1].append(0)
                rgb_avg[2].append(0)

            chest_depths.append(0)
            depth_avg.append(0)
            all_ears.append(0)
            fbf_depths[:, frame_count] = 0
            fbf_ints[:, frame_count] = 0
            frame_count += 1

        intensity_avg = np.array(intensity_avg)
        depth_avg = np.array(depth_avg)
        rgb_avg = np.array(rgb_avg)
        chest_depths = np.array(chest_depths)
        all_ears = np.array(all_ears)

        return intensity_avg, depth_avg, rgb_avg, chest_depths, all_ears, fbf_depths, fbf_ints, frame_count


    



    def get_pixels_in_ROI(self, b_pixels, h, w):
        # Dummy: returns all pixels in bounding box
        x1, y1, x2, y2 = b_pixels
        return [(x, y) for x in range(x1, x2) for y in range(y1, y2)]

    def intensity_to_grayscale(self, pixels):
        # Convert RGB to grayscale
        return 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]

    def smooth_facial(self, x, window_len=20):
        if len(x) < window_len:
            return x
        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        w = np.ones(window_len, 'd')
        y = np.convolve(w/w.sum(), s, mode='valid')
        return y[(window_len//2-1):-(window_len//2)]

    def eye_aspect_ratio(self, eye):
        # Dummy: returns 0, replace with your EAR calculation
        return 0

    def detect_blinks_and_perclos(self, landmarks_pixels):
        # Dummy: returns 0, replace with your blink/perclos logic
        return 0

    def get_face_pixels_from_landmarks(self, landmarks, frame):
        # Dummy: returns all pixels in the frame as the ROI
        # Replace with logic to extract pixels inside the face mesh
        h, w, _ = frame.shape
        return frame.reshape(-1, 3)
    
    def extract_chest_depth(self, depth_frame, roi_coords):
        # roi_coords: (x1, y1, x2, y2)
        x1, y1, x2, y2 = roi_coords
        chest_region = depth_frame[y1:y2, x1:x2]
        if chest_region.size == 0:
            return 0
        return float(np.mean(chest_region))
    
    def draw_bounding_boxes(self, bounding_box_pixels, frame, roi_name):
    # bounding_box_pixels: (x1, y1, x2, y2)
        color = (0, 255, 0)  # Green box
        thickness = 2
        x1, y1, x2, y2 = [int(np.round(v)) for v in bounding_box_pixels]
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, roi_name, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
    
    def get_face_bounding_box(self, landmarks):
    # landmarks: shape (N, 2) for N facial points
        x_coords = [pt[0] for pt in landmarks]
        y_coords = [pt[1] for pt in landmarks]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        return (x1, y1, x2, y2)
    
    def get_bounding_box(self, roi_name, landmarks_pixels):
        face_roi_definitions = {
            'nose': np.array([196, 419, 455, 235]),
            #'nose': np.array([3, 195, 281, 275, 1, 44, 220, 134]),
            'forehead': np.array([109, 338, 9]),
            'right_cheek': np.array([436, 346 , 343, 344]),
            'left_cheek': np.array([129, 121 , 117, 216]), 
            'cheek_n_nose': np.array([117, 346, 411, 187]),
            'low_forehead': np.array([108, 337, 8]),
            'left_eye': np.array([33, 160, 159, 158, 133, 153, 145, 144]),
            'right_eye': np.array([263, 387, 386, 385, 362, 380, 374, 373])
        }
        landmark_indices = face_roi_definitions[roi_name]
        bounding_box_pixels = landmarks_pixels[landmark_indices]
        return bounding_box_pixels



