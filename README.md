# PushUpCounter
A simple program using Mediapipe and OpenCV to count the number of Push Ups done. The main goal is to ensure proper form while doing Push Ups so as to achieve maximum effect. 

You may use the BasicPoseModule in your personal projects, changing the variables as necessary. The Pose Module is using mediapipe's Pose module. Refer to the image below for the different joints in the body that are detected.

![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

## Install Python
Install a recent version of Python.
### macOS
```bash
$ brew install python3
```
### WSL or Linux
```bash
$ sudo apt-get update
$ sudo apt-get install python3 python3-pip python3-venv python3-wheel python3-setuptools
```
## Create a Python virtual environment
This section will help you install the Python tools and packages locally, which won’t affect Python tools and packages installed elsewhere on your computer.

After finishing this section, you’ll have a folder called `env/` that contains all the Python packages you need for this project.

**Pitfall:** Do not use the version of Python provided by Anaconda. 

Create a virtual environment in your project’s root directory. 
```bash
$ pwd
/mnt/c/Users/gurish/OneDrive/Documents/UofM_Clubs/MECC/NoahsArc
$ python3 -m venv env
```
Activate virtual environment. You’ll need to do this **every time** you start a new shell.
```bash
$ source env/bin/activate
```

## TODO

* High knees
* Jumping Jacks: https://medium.com/@ammarrizwan617/a-jumping-jacks-exercise-trainer-9648eb1195cf
* Squats
* Lunges
* Pushups