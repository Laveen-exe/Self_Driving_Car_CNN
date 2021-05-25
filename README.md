Self Driving Car Using Nvidia CNN Architecture<a name="TOP"></a>
===================

- - - - 
# Introduction #

    Self Driving Car using Nvidia CNN Architecture.
    This is the CNN based network made for https://github.com/udacity/self-driving-car-sim simulator provided by Udacity. 
    Used 15k image data to train the model. Did various Data Augmentation methods for better training.









## Model Architecture ##

![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/cnn-architecture-624x890.png)
**The Validation Loss is less than the Training Loss which ia an unusual case, this is because the model is trained on the hard features and the model parameters are validated on rather easy environment (without data augmentation on the images - adding random shadows).



**Graph of LOSS vs Number of Epochs.**
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Loss.png)




*UPDATE .v1*
Trained the model with same architecture with grayscale images.






**Graph of LOSS vs Number of Epochs.**
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Loss_for_grayscale_images.PNG)



**Method 1 trained the model using RGB images**


**Video Showing Udacity Simualator Autonomous mode**
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Simulator.gif)
