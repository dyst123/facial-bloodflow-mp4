from cv2 import fillConvexPoly
import numpy as np
from facemeshmodule_new import FaceMeshDetector
from chestROIReverseEngineering import ChestROI
from scipy.signal import butter, filtfilt
from scipy.spatial import distance as dist

class FaceProcessing:
    def __init__(
       self,
        frames,  # List or numpy array of RGB frames
        depth_frames,  # List or numpy array of depth frames (same length as frames)
        interval_start,
        interval_end,
        intervalMode=False,
        visualize_FaceMesh=False,
        visualize_ROIs=False,
        static_image_mode=False,
        image_width=804,
        image_height=672,
        fps=10,
        getRR=False
    ):
        self.frames = frames
        self.depth_frames = depth_frames
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
        self.chest_intensities = []
        self.chest_depths = []
        self.RR = None
        self.breathCount = None

    def process_batches(self, batch_size=8, visualize_ROI=False, visualize_FaceMesh=False):
        intensity_avg = []
        depth_avg = []
        rgb_avg = [[], [], []]  # R, G, B
        chest_depths = []
        all_ears = []
        fbf_depths = np.zeros((4, self.interval_end - self.interval_start))
        fbf_ints = np.zeros((4, self.interval_end - self.interval_start))
        frame_count = 0

        for idx in range(self.interval_start, self.interval_end):
            frame = self.frames[idx]
            depth_frame = self.depth_frames[idx]
            face_detected, landmarks = self.face_mesh_detector.find_face_mesh(frame, draw=self.visualize_FaceMesh)
            if landmarks is None:
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

            # Example: get bounding box for face ROI
            face_pixels = self.get_face_pixels_from_landmarks(landmarks, frame)
            face_depth = 0
            if face_pixels is not None and len(face_pixels) > 0:
                intensity_avg.append(np.mean(self.intensity_to_grayscale(face_pixels)))
                rgb_avg[0].append(np.mean(face_pixels[:, 0]))
                rgb_avg[1].append(np.mean(face_pixels[:, 1]))
                rgb_avg[2].append(np.mean(face_pixels[:, 2]))
            # --- New: Get face ROI depth ---
            # Get the mask for the face ROI (using landmarks)
                h, w, _ = frame.shape
                face_mask = np.zeros((h, w), dtype=np.uint8)
                # Example: use convex hull of landmarks as mask
                from cv2 import fillConvexPoly
                hull = np.array(landmarks, dtype=np.int32)
                fillConvexPoly(face_mask, hull, (1,))
                face_depth_pixels = depth_frame[face_mask == 1]
                if face_depth_pixels.size > 0:
                    face_depth = np.mean(face_depth_pixels)
                else:
                    face_depth = 0
                depth_avg.append(face_depth)
            else:
                intensity_avg.append(0)
                rgb_avg[0].append(0)
                rgb_avg[1].append(0)
                rgb_avg[2].append(0)
                depth_avg.append(0)

            chest_roi_coords = self.chest_roi._Chest_ROI_extract(frame, landmarks, draw=False)
            if chest_roi_coords is not None:
                # Use the first ROI (index 0)
                roi_corners = chest_roi_coords[0]  # shape (4, 2)
                x_coords = roi_corners[:, 0]
                y_coords = roi_corners[:, 1]
                x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
                y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))
                chest_depth = self.extract_chest_depth(depth_frame, (x1, y1, x2, y2))
                chest_depths.append(chest_depth)
            else:
                chest_depths.append(0)

            # No depth in RGB, so fill with zeros
            

        
        

            # EAR (dummy, replace with your logic)
            all_ears.append(0)

            # FBF (dummy, replace with your logic)
            fbf_depths[:, frame_count] = 0
            fbf_ints[:, frame_count] = 0

            frame_count += 1

        intensity_avg = np.array(intensity_avg)
        depth_avg = np.array(depth_avg)
        rgb_avg = np.array(rgb_avg)
        chest_depths = np.array(chest_depths)
        all_ears = np.array(all_ears)

        print("depth_avg length:", len(depth_avg), "sample values:", depth_avg[:10])

        return intensity_avg, depth_avg, rgb_avg, chest_depths, all_ears, fbf_depths, fbf_ints, frame_count

    def avg_vals(self, pixels, pixels_rgb, d, i, rgb):
        # Example: average over ROI pixels
        avg_intensity = np.mean(i) if len(i) > 0 else 0
        avg_depth = np.mean(d) if len(d) > 0 else 0
        avg_rgb = [np.mean(rgb[:, 0]), np.mean(rgb[:, 1]), np.mean(rgb[:, 2])] if len(rgb) > 0 else [0, 0, 0]
        return avg_intensity, avg_depth, avg_rgb

    def get_bounding_box(self, roi_name, landmarks_pixels):
        # Dummy: returns min/max bounding box for given landmarks
        xs = [p[0] for p in landmarks_pixels]
        ys = [p[1] for p in landmarks_pixels]
        return (min(xs), min(ys), max(xs), max(ys))

    def draw_bounding_boxes(self, bounding_box_pixels, frame, roi_name):
        # Dummy: does nothing, you can use cv2.rectangle if needed
        pass

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