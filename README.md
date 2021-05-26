Self Driving Car Using Nvidia CNN Architecture<a name="TOP"></a>
===================

- - - - 
# Introduction #

    Self Driving Car using Nvidia CNN Architecture.
    This is the CNN based network made for https://github.com/udacity/self-driving-car-sim simulator provided by Udacity. 
    Used 15k image data to train the model. Did various Data Augmentation methods for better training.
    Used Pytorch framework for the model.







## Image Augmentation ##




    Did image resizing and cropping, converted RGB image to YUV
    Performed random flipping and translation to the images for better training.
    
    
    
    
    
```python
    def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


    def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


    def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    
   def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


    def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle
```





## Model Architecture ##








![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/cnn-architecture-624x890.png)















```python
     model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = 3, out_channels = 24,kernel_size = (5,5), stride = (2,2)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3), stride=(1, 1)),
        torch.nn.ELU(),
        torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1, 1)),
        torch.nn.ELU(),
        torch.nn.Dropout(p=args.keep_prob, inplace=True),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features = 1152, out_features = 100,bias = True),
        torch.nn.ELU(),
        torch.nn.Linear(in_features = 100,out_features = 50,bias = True),
        torch.nn.ELU(),
        torch.nn.Linear(in_features = 50,out_features = 10,bias = True),
        torch.nn.ELU(),
        torch.nn.Linear(in_features = 10,out_features = 1,bias = True)
        )
```



## Parameters of the model ##






![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Parmeters.PNG)










## Loss vs Epoch ##






**Graph of LOSS vs Number of Epochs : Method 1 : Trained the model using RGB images with 15k images, Number of Epoch trained upon = 50 training time : 45 minutes**




**The Validation Loss is less than the Training Loss which ia an unusual case, this is because the model is trained on the hard features and the model parameters are validated on rather easy environment (without data augmentation on the images - adding random shadows).








![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Loss.png)



**Running the model**


    Using socketio ran the model drive.py 
    pip install python-engineio==3.13.2
    pip install python-socketio==4.6.1




![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Simulator.gif)


**Video Showing Udacity Simualator Autonomous mode**






*UPDATE .v1*
Trained the model with same architecture with grayscale images. Number of Epoch trained upon = 30, training time = 30 minutes.






**Graph of LOSS vs Number of Epochs.**
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Loss_for_grayscale_images.PNG)






*UPDATE .v2*
Using Image processing and edge detection algorithms, tried to find the edges through the path.






### Image showing output of edge detection on the training images ###



![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Various_edge_detectors.PNG)











### Image showing output of Canny Edge Detector on the training images ###





The Canny edge detection algorithm is composed of 5 steps:


    1. Noise reduction;
    2. Gradient calculation;
    3. Non-maximum suppression;
    4. Double threshold;
    5. Edge Tracking by Hysteresis.
    
    
    
    
    
    
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Various_edge_detectors.PNG)





### Image showing Pixel values vs number of pixels in the image ###






![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Pixel_Values.PNG)

- - - - 

## Conclusion ##



    Can't use edge detection in this model because of lot of noise in the images.
    Roads have some pattern or texture hence not able to remove the noise.

