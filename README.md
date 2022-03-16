The following repo covers a preview of the work I did under Claire Monteleoni and Moumita Saha. The goal of this project was to classify the rainfall type for a given day in advance. We then moved to the final goal, which was to classify **spell detection** (predicting a wet, dry, or normal day). In total, we utilized NOAA's reanalysis data and created a 4-dimensional time series grid stretching from 1948 to 2014. 

Although we used a general pipeline from traditional methods, we ended up creating variants of **Convolutional Neural Networks (CNN).** Above, you can see the code for the models as well as programs to extract the `.netcd4` data. We have also included results for anyone to see how the different variations between the architectures as well as hyperparamaters performed.
