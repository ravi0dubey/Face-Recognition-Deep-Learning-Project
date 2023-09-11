import sys
# from src.insightface.src.common import face_preprocess
from src.insightface.src.common import face_preprocess
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')
from mtcnn.mtcnn import MTCNN #importing mtcnn
# import face_preprocess
import numpy as np
import cv2
import os
from datetime import datetime


# Purpose is the get 50 images from live feed of camera and crop the facial feature of the image and save it in 112*112 dimension
class TrainingDataCollector:

    def __init__(self, args):
        self.args = args
        # Detector = mtcnn_detector
        self.detector = MTCNN()

    def collectImagesFromCamera(self):
        cap = cv2.VideoCapture(0) # initialize video stream
        faces = 0 # Setup some useful var
        frames = 0
        max_faces = int(self.args["faces"]) #max_Faces count set to 50
        max_bbox = np.zeros(4) #bounding box co-ordinate set to 0. Which will be replaced with values coming from MTCNN
        if not (os.path.exists(self.args["output"])): # create dataset/train folder in case if it does not exists
            os.makedirs(self.args["output"])
        # work for each face starting from 1st face till 50    
        while faces < max_faces:
            ret, frame = cap.read() # image from camera is read
            frames += 1
            dtString = str(datetime.now().microsecond) #storing image with name in microsecond
            # Get all faces on current frame
            bboxes = self.detector.detect_faces(frame) #frame captured from camera is passed to detect_faces of mtcnn to get the bounding boxes

            if len(bboxes) != 0: #there should be face which gets detected
                # Get only the biggest face
                max_area = 0
                for bboxe in bboxes:
                    bbox = bboxe["box"]
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) # getting bounding 4 co-ordinates
                    keypoints = bboxe["keypoints"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) # calculating the pixels within the bounding box 
                    if area > max_area:
                        max_bbox = bbox
                        landmarks = keypoints
                        max_area = area

                max_bbox = max_bbox[0:4]

                # get each of 3 frames
                if frames % 3 == 0:
                    # getting the co-ordinates of eyes and nose
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    # cropping the complete image to store only the face and saving in 112*112 dimension
                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')
                    cv2.imwrite(os.path.join(self.args["output"], "{}.jpg".format(dtString)), nimg) #writing the image in train folder
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2) #creating bounding box on the facial image
                    print("[INFO] {} Image Captured".format(faces + 1))
                    faces += 1
            cv2.imshow("Face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
