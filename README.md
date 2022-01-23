# Gender Classification

## Goal
We want to teach the neural network to distinguish man and woman. The results are below: 

 Man            |  Woman
:-------------------------:|:-------------------------:
![](images\man.png)  |  ![](images\woman.png)

## Prerequisites
The following prerequisites need to be installed:
- Python 3.7
- CUDA version 11.5
- CuDNN version 8.2 (as 8.3 have issues to load dlls even though it is CUDA 11.5 compatible)
- Python modules as in requirements.txt
- Graphviz to be able to output your model graphically (version 2.50). Modify the path environment variable to include Graphviz\bin directory. Ideally you want to manage it through some package managers as Winget or Chocolatey.

## Models
There were two models used: one is close to AlexNet architecture, another one is close to VGGNet (MiniVGGNet). The two models are supplied with weights which you can use by keras.loadmodel method (alexnet33.hdf5 and minivggnet10.hdf5).
 AlexNet             |  MiniVGGNet
:-------------------------:|:-------------------------:
![](images\alexnet.png)  |  ![](images\minivggnet.png)

## Datasets
There are multiple datasets available in Kaggle. You can use one of them.

