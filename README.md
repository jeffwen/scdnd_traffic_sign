# Traffic Sign Classification

[//]: # (Source Images)

[internet_images]: ./media/new_internet_images.png "Internet Images"
[aws_curve]: ./media/aws_train_valid_curve.png "AWS Train Curve"

This repository contains an adaptation of the LeNet architecture applied to traffic sign classification. Specifically, the goal was to design, train, validate, and test a convolutional neural network (CNN) architecture while experimenting with various image processing techniques to achieve reasonable results. 

![alt text][internet_images] 

The code for this project is in the [IPython Notebook](https://github.com/jeffwen/sdcnd_traffic_sign/blob/master/Traffic_Sign_Classifier.ipynb) and you can dive into the details of what I did by reading the [write-up](https://github.com/jeffwen/sdcnd_traffic_sign/blob/master/sdcnd_traffic_sign_writeup.md).

The final model (built using Tensorflow) was based on the LeNet architecture with additional dropouts added. The final model achieved:

- Training Set Accuracy: ~99.8%
- Validation Set Accuracy: ~95%
- Test Set Accuracy: ~94%

![alt text][aws_curve] 

The model architecture is as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale/ Normalized image 	   		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU activation		|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU activation       |                                               |
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16  	|
| Fully connected		| 400 input, 120 output     					|
| RELU activation       |                                               |
| Dropout               | 0.6 keep probablility (training)              |
| Fully connected		| 120 input, 84 output     				     	|
| RELU activation       |                                               |
| Dropout               | 0.6 keep probablility (training)              |
| Fully connected		| 84 input, 43 output     				     	|



