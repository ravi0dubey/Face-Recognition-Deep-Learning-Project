
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
3.
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
1. Write **template.p**y which create a folder structure of our project. Within each folders, it will create the filenames where we will be writing our code. </br>
2. Clone **YOLOV5** github repo from git  using "clone https://github.com/ultralytics/yolov5.git" and delete its .git and .gitignore folder </br>

## Theory behind Face Recongition
1. Get input images of the human faces. </br>
2. Human faces needs to be labelled with the name. </br>
3. We need to detect input face and convert into numbers(co-ordinate) </br>![image](https://github.com/ravi0dubey/Face-Recognition-Deep-Learning-Project/assets/38419795/3b1032e8-b053-46a4-9d4a-028a40ed705c)
4. Then we need to do the Facial analysis which is the analysis of image
5. Convert image data into the numbers also called Embeddings
6.  Input image of size 1280 * 720 needs to be cropped to size of 96*96 or 128*128 and then feed to deep learning algorithm.
7.
8.
9. </br>




6.  </br>
