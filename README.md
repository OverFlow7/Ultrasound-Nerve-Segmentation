# Ultrasound-Nerve-Segmentation
This is a TensorFlow implementation of a [U-net](https://arxiv.org/pdf/1505.04597.pdf)-like network for Kaggle competition : "Ultrasound Nerve Segmentation"

The U-net network consist of a contracting path where the inputs are downsampled (like in a standart CNN) followed by and expanding path where they are upsampled.

![alt text](https://github.com/OverFlow7/Ultrasound-Nerve-Segmentation/blob/master/u-net-architecture.png)

A few changes were made from the original design. 

 * Images were reshaped to 128x128 , a power of 2 for convenience in downsampling/upsampling 
 * Dropout was added after each conv/upconv with p=0.6
 * Halved number of features
 * Batch normalization between every layer
 * AdamOptimizer instead of SGD
 
I used cross entropy with class weights as a loss function, to counter class imbalance ( much more black pixels than white pixels)

The best score this model achieved is a dice coefficient of 0.70115 on the private leaderboard,rank 38/973 (Post deadline),  
the competition's winner got 0.73226.

it trained for 55 epochs in about 2 hours and 30 minutes on a Nvidia GTX 960

#How to improve 
 
 * There is a lot of ultrasounds with contradictory masks, doing some pre-processing to either eliminate contradictory masks or replace them, could improve performance 
 * Using bigger size than 128*128 with more features/layers might be better (not enough RAM to test)
 * Some more post-processing to get more realistic masks 
 * Results are a bit inconsistent the score end up being between ~0.695 and ~0.70 , ensembling could prove useful. 
