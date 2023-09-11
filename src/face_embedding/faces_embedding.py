import sys
# from insightface.deploy import face_model
# from insightface.deploy import face_model
# from src.insightface.deploy import face_model
from src.insightface.deploy import face_model

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from imutils import paths # used to get the path where image is stored
import numpy as np
# import face_model
import pickle
import cv2
import os

# Purpose of this class is to convert image into numerical value and saving it in pickel format. This process is called Face Embedding
class GenerateFaceEmbedding:
    def __init__(self, args):
        self.args = args
        self.image_size = '112,112' #image size set to 112*112
        self.model = "./insightface/models/model-y1-test2/model,0" 
        self.threshold = 1.24 # default from insightFace
        self.det = 0 # default from insightFace        

    def genFaceEmbedding(self):
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.args.dataset)) # Grab the paths to the input images in our dataset
        embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det) # Initialize the faces embedder
        # Initialize our lists of extracted facial embeddings and corresponding people names
        knownEmbeddings = []
        knownNames = []

        # Initialize the total number of faces processed
        total = 0

        # Loop over the imagePaths to create the embedding
        for (i, imagePath) in enumerate(imagePaths):            
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2] # extract the person name from the image path
            image = cv2.imread(imagePath) # load the image
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert face to RGB color
            nimg = np.transpose(nimg, (2, 0, 1))
            face_embedding = embedding_model.get_feature(nimg) # Get the face embedding vector
            # add the name of the person + corresponding face embedding to their respective list
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total += 1

        print(total, " faces embedded")
        # save to output
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(self.args.embeddings, "wb") # saving the dictionary in pickle format
        f.write(pickle.dumps(data))
        f.close()
