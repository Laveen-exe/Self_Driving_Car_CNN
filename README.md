Self Driving Car Using Nvidia CNN Architecture<a name="TOP"></a>
===================

- - - - 
# Introduction #

    Self Driving Car using Nvidia CNN Architecture.
    This is the CNN based network made for https://github.com/udacity/self-driving-car-sim simulator provided by Udacity. 
    Used 15k image data to train the model. Did various Data Augmentation methods for better training.
    Used Pytorch framework for the model.










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
```







## Loss vs Epoch ##






**Graph of LOSS vs Number of Epochs : Method 1 : Trained the model using RGB images with 15k images, training time : 45 minutes**




**The Validation Loss is less than the Training Loss which ia an unusual case, this is because the model is trained on the hard features and the model parameters are validated on rather easy environment (without data augmentation on the images - adding random shadows).








![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Loss.png)



**Running the model**


    Using socketio ran the model drive.py 
    pip install python-engineio==3.13.2
    pip install python-socketio==4.6.1




![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Simulator.gif)


**Video Showing Udacity Simualator Autonomous mode**


*UPDATE .v1*
Trained the model with same architecture with grayscale images.






**Graph of LOSS vs Number of Epochs.**
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Loss_for_grayscale_images.PNG)



**Method 1 trained the model using RGB images**


**Video Showing Udacity Simualator Autonomous mode**
![alt text](https://github.com/Laveen-exe/Self_Driving_Car_CNN/blob/main/Media/Simulator.gif)
