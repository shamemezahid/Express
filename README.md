# Express : AI Expression Detection
Express is an AI program that takes the video feed from your devices camera to identify faces and it's expressions using a Convolutional Neural Network. 


### Data Sets used to Train The Model- 
```
https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
https://github.com/afad-dataset/tarball
https://github.com/harish2006/IISCIFD
https://generated.photos/faces/
https://github.com/nikhilroxtomar/Extract-Frame-from-Videos
```

### Usage 
Install the latest version of python from 

Website: https://www.python.org/downloads/ 
(Recommended: v3.10) 

Linux (Debian / Ubuntu / Fedora)
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
```


### Dependency Installation
```
pip install keras tensor-flow opencv-python numpy
```

### Running the Application 
Go to the targetted folder where the ``main.py`` exists, open the path in a commandline interface. 

Windows 11: Right Click Anywhere and Click "Open in Windows Terminal"

Windows 10: Open Powershell, navigate to the target folder using ``cd`` [WikiPedia](https://en.wikipedia.org/wiki/Cd_(command))

Windows <10: Open CMD, navigate to the target folder using ``chdir`` [GeeksForGeeks](https://www.geeksforgeeks.org/cd-cmd-command)

Run the following command to run on windows
```
python.exe .\main.py
```

Linux / MacOS: Open folder in Terminal and run the command 
```
python main.py
```

### Project Summary Documentation 

#### Motivation 
It’s always wise to start with the best. Our team is new to AI coding. So we want to create an app that will teach us the process and pipeline of making an AI app, as well as the joy of programming. Working on this will teach us AI and its application. For several reasons, there are no widespread applications that accurately identify facial expressions. So we worked on it for these reasons.

#### Problem Definition 
A video stream of a person’s face can be used to determine the emotion they are expressing through their face using facial expression recognition. It’s a classification issue for images, because the algorithm must learn to assign an emotion label to each picture. On a large-scale database for face expression recognition, this project was carried out. First, we select an appropriate model for our use case, which requires a large amount of facial expression classification data to feed to our model in order to get reliable detection results. It’s necessary to first load and visualize the image data before we can begin the detecting process. After that, we’ll use machine learning classifiers to identify different facial expressions, so we will see how sample variety influences performance.

