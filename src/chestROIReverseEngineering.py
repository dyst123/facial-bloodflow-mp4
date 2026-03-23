import os
import time
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.io import savemat
# from scipy.io import loadmat
from typing import Tuple, Union
from scipy.spatial import distance as dist
import concurrent

class ChestROI:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    def getFacePoints(self, img):
    # print("in face function")
        mpFaceMesh = mp.solutions.face_mesh #imports faceMesh class # type: ignore
        FaceMesh = mpFaceMesh.FaceMesh() #face mesh object, has various parameters u can find on github
        results = FaceMesh.process(img)

        #print("results:", results)

        height, width, channel = img.shape
        mpDraw = mp.solutions.drawing_utils # type: ignore

        neckx = None
        necky = None
        chinx = None
        chiny = None

        faceLandmarks = []

        if results.multi_face_landmarks: #if detect face landmarks
            #print("in loopspsps")
            for faceLms in results.multi_face_landmarks: #for the landmarks corresponding to each face
                neckx = int(faceLms.landmark[58].x * width)
                necky = int(faceLms.landmark[58].y * height)
                cv2.circle(img, (neckx, necky), 5, (255,0,255), cv2.FILLED)

                chinx = int(faceLms.landmark[152].x * width)
                chiny = int(faceLms.landmark[152].y * height)
                cv2.circle(img, (chinx, chiny), 5, (255,0,255), cv2.FILLED)

                for i in range(468):
                    x = faceLms.landmark[i].x * width
                    y = faceLms.landmark[i].y * height
                    faceLandmarks.append([x,y])

                return faceLandmarks




        

    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int,
                image_height: int) -> Union[None, Tuple[int, int]]:
            """Converts normalized value pair to pixel coordinates."""

            # Checks if the float value is between 0 and 1.
            def is_valid_normalized_value(value: float) -> bool:
                return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

            if not (is_valid_normalized_value(normalized_x) and
                    is_valid_normalized_value(normalized_y)):
                # TODO: Draw coordinates even if it's outside of the image bounds.
                return None
            x_px = min(math.floor(normalized_x * image_width), image_width - 1)
            y_px = min(math.floor(normalized_y * image_height), image_height - 1)
            return x_px, y_px
    
    
    def _ROICalc(self, shoulder_x, shoulder_y, neck_width, neck_height, chest_width, chest_height, ROIType = 0):
         
        landmark_px_rr = np.zeros([2, 4, 2]) #for each of the 2 ROIs, we'll have 4 landmarks, and an x,y coord at each landmark

        ratios = []
        if ROIType == 0:
            ratios = [0.5, 1.1, 0.1, 0.3, 0, 1]

        elif ROIType == 1:
            ratios = [0.5, 1.1, 0.1, 0.3, 1, 2]

        elif ROIType == 2:
            ratios = [0.5, 1.1, 0.1, 0.18, 1, 2]

        elif ROIType == 3:
            ratios = [0.5, 1.1, 0.1, 0.18, 1.75, 2.75 ]


        landmark_px_rr[0, :, 0] = [ #for ROI 1 our 4 x coordinates
            int(shoulder_x - ratios[0] * chest_width),
            int(shoulder_x + ratios[0] * chest_width),
            int(shoulder_x + ratios[0] * chest_width),
            int(shoulder_x - ratios[0] * chest_width)
        ]

        landmark_px_rr[0, :, 1] = [ #for ROI 1 our 4 y coordinates
            int(shoulder_y + ratios[1] * neck_height),
            int(shoulder_y + ratios[1] * neck_height),
            int(shoulder_y + ratios[2] * neck_height),
            int(shoulder_y + ratios[2]* neck_height),
        ]

        # landmark_px_rr[0,:,0]=[chin_location[1][0]-0.8*neck_width,chin_location[3][0],
        #                       chin_location[3][0],chin_location[1][0]-0.8*neck_width]
        # landmark_px_rr[0,:,1]=[chin_location[1][1]+10,chin_location[3][1]+10,
        #                       chin_location[3][1]+neck_height,chin_location[1][1]+neck_height]

        landmark_px_rr[1, :, 0] = [ #for ROI 2 our 4 x coordinates
            int(shoulder_x - ratios[3] * chest_width),
            int(shoulder_x + ratios[3] * chest_width),
            int(shoulder_x + ratios[3] * chest_width),
            int(shoulder_x - ratios[3] * chest_width)
        ]

        landmark_px_rr[1, :, 1] = [ #for ROI 2 our 4 y coordinates
            int(shoulder_y + ratios[4] * chest_height),
            int(shoulder_y + ratios[4] * chest_height),
            int(shoulder_y + ratios[5] * chest_height),
            int(shoulder_y + ratios[5] *chest_height)
        ]

        return landmark_px_rr

            

    def _Chest_ROI_extract(self, image, face_landmarks, draw=False, ROI=3):
           
            #takes in an image in rgb numpy format (normally would be grey), and face_landmarks as a 468 by 2 array from the face mesh
            #where first index gives facial landmark and second index gives x,y
            #returns corners of 2 chest ROIs as a 2 by 4 by 2 array where first index gives ROI, second index gives corner number and 3rd index gives x,y coord
            
            #image_3chnl = np.stack((image,) * 3, axis=-1) #image is given as grayscale, so these 2 lines convert to RGB for sake of processing
            #image_3chnl = cv2.convertScaleAbs(image_3chnl)

            image_3chnl = image

            #cv2.imshow("image", image_3chnl)
            #cv2.waitKey(1000)

            shoulder_landmark = [11, 12] #mediapipe pose library has shoulders landmarked at 11,12
            landmark_px_rr = np.zeros([2, 4, 2]) #for each of the 2 ROIs, we'll have 4 landmarks, and an x,y coord at each landmark
            image_height, image_width, channel = image.shape
            results = self.pose.process(image_3chnl) #gets our result

        # print(results)
            #print(results.pose_landmarks)

            body_points = results.pose_landmarks.landmark
            #shoulder_point_l = _normalized_to_pixel_coordinates(body_points[11].x, body_points[11].y, image_width = image_width, image_height = image_height) #left shoulder coordinate
            #shoulder_point_r = _normalized_to_pixel_coordinates(body_points[12].x, body_points[12].y, image_width, image_height) #right shoulder coordinate
            shoulder_point_l = (min(math.floor(body_points[11].x * image_width), image_width - 1) , min(math.floor(body_points[11].y * image_height), image_height - 1))
            shoulder_point_r = (min(math.floor(body_points[12].x * image_width), image_width - 1) , min(math.floor(body_points[12].y * image_height), image_height - 1))

            shoulder_x = (shoulder_point_l[0] + shoulder_point_r[0]) / 2 # x midpoint of shoulders 
            shoulder_y = (shoulder_point_l[1] + shoulder_point_r[1]) / 2 # y midpoint of shoulders

            #image_3chnl, chinx, chiny, neckx, necky = self.getFacePoints(image_3chnl)

            chinx = face_landmarks[58][0]
            chiny = face_landmarks[58][1]

            neckx = face_landmarks[152][0]
            necky = face_landmarks[152][1]

            neck_width = abs(chinx-neckx) * 2
            neck_height = abs(shoulder_y - chiny) * .5

            chest_width = np.abs(shoulder_point_l[0] - shoulder_point_r[0]) #chest width just distance from left to right shoulder
            chest_height = 0.22 * chest_width #chest height is some scale factor of chest width

            #calculations using shoulders and neck to outline chest region
            landmark_px_rr = self._ROICalc(shoulder_x, shoulder_y, neck_width, neck_height, chest_width, chest_height, ROIType=ROI)


            # landmark_px_rr[1,:,0]=[shoulder_point_l[0]-25,shoulder_point_r[0]+25,
            #                       shoulder_point_r[0]+25,shoulder_point_l[0]-25]
            # landmark_px_rr[1,:,1]=[shoulder_point_l[1],shoulder_point_r[1],
            #                       shoulder_point_r[1]+chest_height,shoulder_point_l[1]+chest_height]

            
            #ensures all array coordinates are within image values
            np.clip(landmark_px_rr[0, :, 0], 0, image_width)
            np.clip(landmark_px_rr[0, :, 1], 0, image_height)
            np.clip(landmark_px_rr[1, :, 0], 0, image_width)
            np.clip(landmark_px_rr[1, :, 1], 0, image_height)


            #print("landmark 1:", landmark_px_rr[0][0][0], landmark_px_rr[0][0][1])

            if draw == True:
                cv2.circle(image_3chnl, (int(landmark_px_rr[0][0][0]), int(landmark_px_rr[0][0][1])), 5, (255,0,255), cv2.FILLED)
                cv2.circle(image_3chnl, (int(landmark_px_rr[0][1][0]), int(landmark_px_rr[0][1][1])), 5, (255,0,255), cv2.FILLED)
                cv2.circle(image_3chnl, (int(landmark_px_rr[0][2][0]), int(landmark_px_rr[0][2][1])), 5, (255,0,255), cv2.FILLED)
                cv2.circle(image_3chnl, (int(landmark_px_rr[0][3][0]), int(landmark_px_rr[0][3][1])), 5, (255,0,255), cv2.FILLED)

                cv2.line(image_3chnl, (int(landmark_px_rr[0][0][0]), int(landmark_px_rr[0][0][1])), (int(landmark_px_rr[0][1][0]), int(landmark_px_rr[0][1][1])), (0,0,0), 2) 
                cv2.line(image_3chnl, (int(landmark_px_rr[0][1][0]), int(landmark_px_rr[0][1][1])), (int(landmark_px_rr[0][2][0]), int(landmark_px_rr[0][2][1])), (0,0,0), 2) 
                cv2.line(image_3chnl, (int(landmark_px_rr[0][2][0]), int(landmark_px_rr[0][2][1])), (int(landmark_px_rr[0][3][0]), int(landmark_px_rr[0][3][1])), (0,0,0), 2) 
                cv2.line(image_3chnl, (int(landmark_px_rr[0][3][0]), int(landmark_px_rr[0][3][1])), (int(landmark_px_rr[0][0][0]), int(landmark_px_rr[0][0][1])), (0,0,0), 2) 
                



                cv2.circle(image_3chnl, (int(landmark_px_rr[1][0][0]), int(landmark_px_rr[1][0][1])), 5, (255,0,255), cv2.FILLED)
                cv2.circle(image_3chnl, (int(landmark_px_rr[1][1][0]), int(landmark_px_rr[1][1][1])), 5, (255,0,255), cv2.FILLED)
                cv2.circle(image_3chnl, (int(landmark_px_rr[1][2][0]), int(landmark_px_rr[1][2][1])), 5, (255,0,255), cv2.FILLED)
                cv2.circle(image_3chnl, (int(landmark_px_rr[1][3][0]), int(landmark_px_rr[1][3][1])), 5, (255,0,255), cv2.FILLED)


                cv2.line(image_3chnl, (int(landmark_px_rr[1][0][0]), int(landmark_px_rr[1][0][1])), (int(landmark_px_rr[1][1][0]), int(landmark_px_rr[1][1][1])), (0,0,0), 2) 
                cv2.line(image_3chnl, (int(landmark_px_rr[1][1][0]), int(landmark_px_rr[1][1][1])), (int(landmark_px_rr[1][2][0]), int(landmark_px_rr[1][2][1])), (0,0,0), 2) 
                cv2.line(image_3chnl, (int(landmark_px_rr[1][2][0]), int(landmark_px_rr[1][2][1])), (int(landmark_px_rr[1][3][0]), int(landmark_px_rr[1][3][1])), (0,0,0), 2) 
                cv2.line(image_3chnl, (int(landmark_px_rr[1][3][0]), int(landmark_px_rr[1][3][1])), (int(landmark_px_rr[1][0][0]), int(landmark_px_rr[1][0][1])), (0,0,0), 2) 
                



                cv2.imshow("image", image_3chnl)
                cv2.waitKey(1)

            return landmark_px_rr


"""img = cv2.imread("pose1.jpg")

_Chest_ROI_extract(img)
"""

def main():

    chest = ChestROI()
    cap = cv2.VideoCapture("videoPose.mp4")

    while True:
        success, img = cap.read()
                
        if (not success):
            break
        
        faceLandmarks = chest.getFacePoints(img)
        
        chest._Chest_ROI_extract(img, faceLandmarks, draw = True)


if __name__ == "__main__":
    main()
     