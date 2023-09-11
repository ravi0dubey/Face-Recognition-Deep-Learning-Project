
# Face-Recognition-Deep-Learning-Project

## Problem Statement

We need a solution where in addition to detecting human face, model should be able to verify whose face it is.


## Solution Proposed

In this project, the focus is to correctly detect the face and identify the face of the user using MTCNN model


## Tech Stack Used
Python </br>
MTCNN</br>




# How to run the project
Step 1 : open your anaconda prompt (for windows user search inside start menu )
                                   (for Ubuntu and Mac user you can open your terminal)

Step 2 : Create a new environment
                command : conda create -n facerecognition python==3.6.9 -y </br>
                
Step 3 : activate your environment  </br>
                conda activate facerecognition  </br>
Step 4 : conda install -c anaconda mxnet </br>

Step 5 : conda install -c conda-forge dlib </br>

Step 6 : Uninstall existing version of numpy and install numpy 1.16.1 version: </br>
        pip uninstall numpy </br>
        pip uninstall numpy </br>
        pip install numpy==1.16.1 </br>

Step 7:  Install requirements.txt in the newly created environment</br>
         pip install -r requirements.txt</br>

Step 8 : Installation and setup is done:</br>
         a).  cd src</br>
         b). python app.py</br>

## Video link of project demo


## How project was designed and build
1. Write **template.p**y which create a folder structure of our project. Within each folders, it will create the filenames where we will be writing our code. </br>
2. Clone **YOLOV5** github repo from git  using "clone https://github.com/ultralytics/yolov5.git" and delete its .git and .gitignore folder </br>
