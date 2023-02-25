<h1>Sports Ball detection on Yolov5 Using Jetson Nano 2gb</h1>

<h2>Aim And Objectives :-</h2>

<b>Aim :-</b>

To create a Sports Balls detection system which will detect objects based on whether it is baseball, basketball, american_football, cricket ball, tennis ball, football or golf ball.

<b>Objectives :-</b>

* The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

* Using appropriate datasets for recognizing and interpreting data using machine learning.

* To show on the optical viewfinder of the camera module whether a sports balls belongs to which species.

<h3>Abstract :-</h3>

* Many  computer  vision  applications  rely  on accurate and fast object detection, and in our case, ball detection serves  as  a  prerequisite  for  action  recognition  in  handball scenes.

* We have completed this project on jetson nano which is a very small computational device.

* A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

* One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

* The purpose of this project is to observe which sports balls in showing.

<h3><b>Introduction :-</b></h3>

* We compare the performance of two of the state-of-the-art convolutional neural network-based object detectors for the task of ball detection in non-staged, real-world conditions. The comparison  is  performed  in  terms  of  speed  and  accuracy measures on a dataset comprising custom handball footage and a  sample  of  images  obtained  from  the  Internet.  The performance  of  the  models  is  compared  with  and  without additional training with examples from our dataset.

* Neural networks and machine learning have been used for these tasks and have obtained good results.

* Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Sports ball detection as well.

<h2>Jetson Nano Compatibility :-</h2>

* The power of modern AI is now available for makers, learners, and embedded developers everywhere.

* NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

* Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

* NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

* In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

<h2>Jetson Nano 2GB:-</h2>
![image](https://user-images.githubusercontent.com/119781014/221340609-5048f6d3-a312-4e66-9eca-c64b3da1fed2.png)

## Proposed System :-

1] Study basics of machine learning and image recognition. 

2]Start with implementation • Front-end development • Back-end development 

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether the balls belongs to which spices. 

4] Use datasets to interpret the object and suggest whether the balls on the camera’s viewfinder belongs to which spices.

## Methodology :- 

* A ball can be made from many different materials, leather, rubber, and synthetics being most common in modern times. However, balls made from indigenous materials, particularly from animal parts, were once the norm.

* Some form of ball game is portrayed on early Egyptian monuments. Even among the Romans, who disliked participatory sports, ball play was extremely popular. The Roman baths set aside apartments for ball play, and many gentlemen had ball courts in their private villas.

* Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

* There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

* YOLOv5 was used to train and test our model for whether the sports balls belongs to which species. We trained it for 110 epochs and achieved an accuracy of approximately 92%.

## Installation

### **Initial Setup**

Remove unwanted Applications. sudo apt-get remove --purge libreoffice* sudo apt-get remove --purge thunderbird* Create Swap file sudo fallocate -l 10.0G /swapfile1 sudo chmod 600 /swapfile1 sudo mkswap /swapfile1 sudo vim /etc/fstab #################add line########### /swapfile1 swap swap defaults 0 0 Cuda Configuration vim ~/.bashrc #############add line ############# export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P ATH}} export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 source ~/.bashrc Udpade a System sudo apt-get update && sudo apt-get upgrade ################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1############################# sudo apt install curl curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py sudo python3 get-pip.py sudo apt-get install libopenblas-base libopenmpi-dev source ~/.bashrc sudo pip3 install pillow curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl sudo python3 -c "import torch; print(torch.cuda.is_available())" Installation of torchvision. git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision cd torchvision/ sudo python3 setup.py install Clone yolov5 Repositories and make it Compatible with Jetson Nano. cd git clone https://github.com/ultralytics/yolov5.git cd yolov5/ sudo pip3 install numpy==1.19.4 history ##################### comment torch,PyYAML and torchvision in requirement.txt################################## sudo pip3 install --ignore-installed PyYAML>=5.3.1 sudo pip3 install -r requirements.txt sudo python3 detect.py sudo python3 detect.py --weights yolov5s.pt --source 0

### Sports Balls Dataset Training

* We used Google Colab And Roboflow

* Train your model on colab and download the weights and paste them into yolov5 folder link of project

* Running Fish Species Detection Model source '0' for webcam !python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

### Demo link:-

### Advantages

* High accuracy of sports balls detection

* Less human intervention

### Future Scope

* As we know technology is marching towards automation, so this project is one of the step towards automation.
* Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.
* Garbage segregation will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the situation in an efficient way.
* As more products gets released due to globalization and urbanization new waste will be created and hence our model which can be trained and modified with just the addition of images can be very useful.

### Conclusion

• In this project our model is trying to detect objects and then showing it on viewfinder, live as what their class is as whether they are baseball, basketball, american_football, cricket ball, tennis ball, football or golf ball as we have specified in Roboflow.

• Thus the obtained datasets were preprocessed by using deep learning and yolov5 as tool to detect the sports balls in the datasets were detected and the accuracy was displayed. The type of sports balls were displayed.

• This balls recognition and detection can be used to detect the sports balls, recognize the species in the sports balls play ground and also for commercial purposes.

• Being implemented in real time it can be enhanced by use of high end underwater cameras to get more accuracy in the output.

### References

* Roboflow :- https://roboflow.com/
* Datasets or images used: https://www.gettyimages.ae/photos/sports-balls
* Google images
