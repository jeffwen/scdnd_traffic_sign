## Traffic Sign Recognition

In this project, we classify [German traffic signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

The objective of this project is to ultimately design, train, and test a convolutional neural netowk (CNN) to see if we can accurately classify the traffic signs. Specifically, with image classification tasks the variability of input images might cause the model to perform poorly. In order to remedy this situation, we will consider preprocessing steps to ensure that the model is robust to varying input images. Furthermore, depending on how the model performs, we may need to consider regularization steps such as dropouts or L2 regularization. 

To start with, I use the [LeNet-5 architecture](http://yann.lecun.com/exdb/lenet/) that was first published by Yann Lecun's lab in 1998. 

[//]: # (Example Images)

[example1]: ./media/example1.png "80 kph"
[example2]: ./media/example2.png "Straight or right"
[example3]: ./media/example3.png "Stop"
[example4]: ./media/example4.png "100 kph"
[example5]: ./media/example5.png "30 kph"
[example6]: ./media/example6.png "Beware ice/ snow"
[stop1]: ./media/stop1.png "Stop Sign"
[stop2]: ./media/stop2.png "Stop Sign"
[stop3]: ./media/stop3.png "Stop Sign"
[stop4]: ./media/stop4.png "Stop Sign"
[stop5]: ./media/stop5.png "Stop Sign"
[stop6]: ./media/stop6.png "Stop Sign"
[histogram]: ./media/histogram.png "Class Histogram"
[lenet-5]: ./media/lenet.png "LeNet Architecture"
[before1]: ./media/before_process.png "Before Processing"
[before2]: ./media/before_process2.png "Before Processing"
[after1]: ./media/after_process.png "After Processing"
[after2]: ./media/after_process2.png "After Processing"
[train_valid_original]: ./media/train_valid_original.png "Train vs Validation Curve (Original)"
[train_valid_gray_norm]: ./media/train_valid_gray_norm.png "Train vs Validation Curve (Grayscale, Normalization, Dropout)"
[actual_vs_pred_hist]: ./media/actual_vs_pred_hist.png "Actual vs. Predicted Histogram"
[gray_norm_confusion]: ./media/gray_norm_confusion.png "Confusion Matrix (grayscale, normalization, dropounts)"
[class16]: ./media/class16_class41_2.png "Class 16"
[class41]: ./media/class16_class41_1.png "Class 41"
[class0]: ./media/class0_class4_1.png "Class 0"
[class4]: ./media/class0_class4_2.png "Class 4"
[new1]: ./media/class0_class4_2.png "Class 4"
[new2]: ./media/class0_class4_2.png "Class 4"



### Data Set Summary & Exploration

Before jumping into the model, we can take a look at some of the images within the dataset. 

![alt text][example1] ![alt text][example2] ![alt text][example3] 
![alt text][example4] ![alt text][example5] ![alt text][example6]

The images vary quite a lot in terms of brightness/ we can imagine that the orientation of the images might also be quite different. More specifically, we can take a look at just stop signs and see that there is quite a lot of variation. 

![alt text][stop1] ![alt text][stop2] ![alt text][stop3] 
![alt text][stop4] ![alt text][stop5] ![alt text][stop6]

Additionally, we can take a look at how the traffic sign classes are distributed (to see if the classes are evenly balanced). Looking at the below histogram, it seems like there are some classes that are represented 7x as much as the least  represented class. Later on we might want to upsample the less represented classes to make the model more robust. 
![alt text][histogram]

Below are some summary statistics calculated from the dataset that we have:

- Number of training examples = `34799`
- Number of validation examples = `4410`
- Number of testing examples = `12630`
- Image data shape = `(32, 32)`
- Number of classes = `43`

### Data Preprocessing and Feature Engineering

As evidenced above, the input data is rather varied in terms of lighting and orientation of images. In order to improve the accuracy of the model, we can implement some preprocessing steps on our images. 

First of all, I implemented grayscaling of the traffic sign images. While there may be some information lost because the color information in the image might contain information about the traffic sign, grayscaling simplifies the 3 channels into a single channel, which can lead to quicker model training. Additionally, taking away the color channels might allow the CNN to "focus" on extracting the important non-color features better. 

After performing grayscaling, I performed mean subtraction and normalization of the image pixels. Specifically, in the cs231n [course notes](http://cs231n.github.io/neural-networks-2/) Andrej Karpathy mentions that the standard preprocessing techniques involve mean subtraction `img = img - np.mean(img)` and normalization `img = img / np.std(img)`. 

Before Processing Images
![alt text][before1] ![alt text][before2] 

After Processing Images
![alt text][after1] ![alt text][after2] 

At this point, model performed fairly well and better than the 93% on the validation set that was required in the grading rubric. To start with, the original LeNet-5 network applied to the colored images achieved ~88% on the validation set. Then the gray scaling, mean subtraction, and normalization combined with dropouts on the fully connected layers pushed the validation accuracy to ~95%. 
- I added the dropouts because looking at the original train vs. validation curve, the model shows strong overfitting. As to why the drops were only applied to the fully connected layers, looking at this [kaggle question](https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/20201) the author mentions that we usualy do not apply dropouts to the convolution layer because the convolution layer is meant to extract features and also does not have that many features to regularize.

Original LeNet Architecture
![alt text][train_valid_original] 

LeNet Architecture with Grayscaling, Normalization, and Dropouts
![alt text][train_valid_gray_norm]

While the model is performing fairly well, we can dive deeper in to the performance of the model to figure out what is happening with the predictions. Specifically, below are the histogram and confusion matrix of validation actuals vs. the validation predicted. By looking at the histogram and confusion matrix, we can look at where the classifications are mistaken and what classes are typically wrong. 

![alt text][actual_vs_pred_hist]

![alt text][gray_norm_confusion]

If we look at th histogram above, we see that most of the bars are at about the same place. Similarly, it seems like the diagonal is quite well defined, which means that the CNN did fairly well! However, if we look at the off diagonal cells in the confusion matrix, we see that there are definitely misclassifications. For example, what is actually supposed to be class 0 gets misclassified as class 4 or class 16 gets misclassified as class 41. Below are sample images from these classes.

Class 0 misclassified as Class 4
![alt text][class0] ![alt text][class4]

Class 16 misclassified as Class 41
![alt text][class16] ![alt text][class41]

So it seems fairly reasonable, especially for the class 0 to class 4 example for the network to misclassify the images. It even looking at the gray and normalized images the differences between the two are not that drastic. In order to improve the model's performance on these, we can augment the dataset by creating more images with various operations like zooming and rotating the images. Note that it seems like the classes that are more often misclassified are also the classes with the fewest amount of data. This step took an unexpectedly long time because many additional images were created of the under represented classes. With these additional images, we retrained the model. Overall, 11915 additional images were created and a sample of the additional images are shown below.

![alt text][class16] ![alt text][class41]



### Design and Test a Model Architecture

With an idea of how the data looks, we can start to design the architecture of the CNN to classify these traffic signs. Note that during the training and validation steps we will revisit the preprocessing of the data to make sure that the model improves/ does not under or overfit. 

In the first iteration of the CNN, I use a modified version of the LeNet-5 architecture (shown below). The code used to build the CNN is under the "Model Architecture" section of the Jupyter Notebook. 

![alt text][lenet-5]

Specifically, the model had the following setup:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU activation		|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU activation       |                                               |
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16  	|
| Fully connected		| 400 input, 120 output     					|
| RELU activation       |                                               |
| Fully connected		| 120 input, 84 output     				     	|
| RELU activation       |                                               |
| Fully connected		| 84 input, 43 output     				     	|



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
