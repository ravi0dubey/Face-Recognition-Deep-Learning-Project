
# Face-Recognition-Deep-Learning-Project

## Problem Statement

We need a solution where in addition to detecting human face, model should be able to verify whose face it is.


## Solution Proposed

In this project, the focus is to correctly detect the face and identify the face of the user using deepinsight/InsightFace.
InsightFace is an integrated Python library for 2D&3D face analysis. It efficiently implements a rich variety of state of the art algorithms of face recognition, face detection and face alignment, which optimized for both training and deployment. </br>
**github link of InsightFace** https://github.com/deepinsight/insightface

## Tech Stack Used
1. Python </br>
2. MTCNN(Multi-task Cascaded Convolutional Networks)  https://pypi.org/project/mtcnn/  </br>
3.ss
4.
5.
6.




# How to run the project
**Step 1** : open your anaconda prompt (for windows user search inside start menu )
                                   (for Ubuntu and Mac user you can open your terminal)

**Step 2** : Create a new environment
                command : conda create -n facerecognition python==3.6.9 -y </br>
                
**Step 3** : activate your environment  </br>
                conda activate facerecognition  </br>
**Step 4** : conda install -c anaconda mxnet </br>

**Step 5** : conda install -c conda-forge dlib </br>

**Step 6** : Uninstall existing version of numpy and install numpy 1.16.1 version: </br>
        pip uninstall numpy </br>
        pip uninstall numpy </br>
        pip install numpy==1.16.1 </br>

**Step 7**:  Install requirements.txt in the newly created environment</br>
         pip install -r requirements.txt</br>

**Step 8** : Installation and setup is done:</br>
         a).  cd src</br>
         b). python app.py</br>


## Video link of project demo


## How project was designed and build
1. **app.py->** Driver program of the project which invokes the camera and then call subsquent method from each modules to perform the operations of collecting pictures from camera,training it and prediction of the face . </br>
2. **get_faces_from_camera.py->** Purpose is the get the 50 images from live feed of camera and crop the facial feature of the image and save it in 112*112 dimension </br> 
3. **faces_embedding.py->** Purpose of this class is to convert image into numerical value and saving it in pickel format. This process is called Face Embedding </br>
4. **train_softmax.py->** Purpose is to train the model using embeddings of the image. Model is trained in batchsize of 8 with 5 epochs. Relu activation for hidden layer and softmax for output layer. Saving the output as pickle format.</br>
5. **facePredictor.py->** Purpose is to do the prediction of the face. </br>



## Theory behind Face Recongition
1. Get input images of the human faces. </br>
2. Human faces needs to be labelled with the name. </br>
3. We need to detect input face
4. Input image of size 1280 * 720 needs to be cropped to size of 96*96 or 128*128 and then feed to deep learning algorithm.
5. MTCNN detects the bounding box co-ordinates, co-ordinates of keypoints of the face(nose, mouth-right,right-eye,left-eye,mouth_left) and the confidence score of the face image)
6. Then we need to do the Facial analysis, we need to create small feature and create array of the features.
7. Converting image data into the numbers also called Embeddings ![image](https://github.com/ravi0dubey/Face-Recognition-Deep-Learning-Project/assets/38419795/3b1032e8-b053-46a4-9d4a-028a40ed705c)
8. ds
9. ss
10. ss
11. ss
12.
13. </br>




6.  </br>
