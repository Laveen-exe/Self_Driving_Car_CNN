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



- - - - 

## Model Architecture ##








![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/cnn-architecture-624x890.png)













- - - - 

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


- - - - 
## Parameters of the model ##






![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/Parameters.PNG)








- - - - 

## Loss vs Epoch ##






**Graph of LOSS vs Number of Epochs : Method 1 : Trained the model using RGB images with 15k images, Number of Epoch trained upon = 50 training time : 45 minutes**




**The Validation Loss is less than the Training Loss which ia an unusual case, this is because the model is trained on the hard features and the model parameters are validated on rather easy environment (without data augmentation on the images - adding random shadows).








![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/Loss.png)


- - - - 
**Running the model**


    Using socketio ran the model drive.py 
    pip install python-engineio==3.13.2
    pip install python-socketio==4.6.1




![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/Simulator.gif)

- - - - 
**Video Showing Udacity Simualator Autonomous mode**






*UPDATE .v1*
Trained the model with same architecture with grayscale images. Number of Epoch trained upon = 30, training time = 30 minutes.





- - - - 
**Graph of LOSS vs Number of Epochs.**
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/Loss_for_grayscale_images.PNG)






*UPDATE .v2*
Using Image processing and edge detection algorithms, tried to find the edges through the path.





- - - - 
### Image showing output of edge detection on the training images ###



![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/Various_edge_detectors.PNG)










- - - - 
### Image showing output of Canny Edge Detector on the training images ###





The Canny edge detection algorithm is composed of 5 steps:


    1. Noise reduction;
    2. Gradient calculation;
    3. Non-maximum suppression;
    4. Double threshold;
    5. Edge Tracking by Hysteresis.
    
    
    
    
    
    
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/Various_edge_detectors.PNG)




- - - - 
### Image showing Pixel values vs number of pixels in the image ###






![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/Pixel_Values.PNG)

- - - - 

## Conclusion ##



    Can't use edge detection in this model because of lot of noise in the images.
    Roads have some pattern or texture hence not able to remove the noise.
    
    
- - - - 
    
# Introducing Dynamic Obstacles #


 ## Model Architecture ##
   ![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/media_files/model_architecture_with_central_camera.PNG)
    
- - - -     
   #### Camera Views

<table>
  <tr>
    <td>Left Camera View</td>
     <td>Right Camera View</td>
     <td>Central Camera View</td>
  </tr>
  <tr>
    <td><img src="media_files/left_with_central_camera.PNG" width=350 height=400></td>
    <td><img src="media_files/right_with_central_camera.PNG" width=350 height=400></td>
    <td><img src="media_files/central_with_central_camera.PNG" width=350 height=400></td>
  </tr>
 </table>

   ## Approach 1 ##
   Approach 1 is to add  8-9 dynamically moving objects at random positions, which moves randomly (used random function to generate some random points under some threshold for     
   every 5ms gap). It didn't work, the car was crashing with the obstacles. Rather than trying with different architectures we tried different approach.
   
   **PROS**
   1. The car (moving agent) is able to avoid some of the obstacles while crashing into other, we were not sure about the overfitting whether it is actually avoiding or just is a 
   random chance of luck and it is moving on that path which we gathered during training.
   2. For testing this overfitting thing we tried to move the car (moving agent) on the reversed track. For our surprise it worked although not properly, it's still crashing but 
   it avoid some of the objects very smoothly.
   
  **CONS**
   1. The car is crashing!!! maybe the data we collected is not enough, or the architecture is not good enough or the model is learning the position of the objects rather than the 
   ability to avoid them. So we tried Approach 2 to add more obstacles at fixec positions.

   ## Approach 2 ##
   We collected points (around 35-40) on each side of the entire track (corresponding left and right points) with some distance between them and added cubes of size 3x3 then          allowed them to move on a straight path to the corresponding opposite point (if the object is at left side then it will move to right side of the track and other way). We          trained this model by switching on the obstacles when the obstacle is at some threshold distance form the car. We experimented with different with different architectures,
   but it's not able to detect obstacles which are infront of the car. 
    
   ## Approach 3 ##
   To solve above issue we added central camera which covers front part of the car and trained the model with input as 3 channels (gray scale images of left, right and central        images). 
    
   ## Approach 4 ##
   ( this approach is yet to implement )
   This approach is for randomising the positions of all obstacles.
   The idea is to generate the obstacles randomly when the car is at some threshold radius without any predefined points (the points should change for each lap at each location)      and move them radially (line joining centre of whole track and point where object is present) and after the car crosses these obstacles, they should disappear. So in entire        track we will have obstacles only in some area where the car is present.

