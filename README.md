# NSDropout (Neuron Specific Dropout)

## NOTE ##
The first two files, New_Dropout.ipynb and Old_Dropout.ipynb, were used for development of initial layer and do not represent the accuracy of the most curent version.

## Error rate (%) ##
   &#xfeff;             | MNIST Numbers | MNIST Fashion
-------------           | ------------- | -------------
Old Dropout             | 5.59          | 15.23
New Dropout             | 0.00*         | 0.89
Highest Reported[1] [2] | 0.13          | 3.09

Yes there are smaller MNIST error rates using CNNs, data augmentation, preprocessing ect with a normal dropout layer but in this test the only variables changed were the dropout layer. This aloud for a direct comparison between layers and the improvments the new layer made.

 _*No images were miss-classified. Model trained on 9,600 images a validaiton on 10,000. Testing images were split up 8000 for optimization and 1600 for new dropout layer. See mnist_numbers_implementation_of_New_Dropout.ipynb for more information._

## Testing Methodology ##

With saving the model from the best epoch not set up I let every model run for 1000 epochs(both old dropout models' validation accuracy was decreasing at this point) and recorded the epoch where it's validation accuracy was the highest. Thanks to setting the numpy seed I was able to run the training again and stop at the epoch where the model previously hit it's high.

## To-Do ##

- [X] Partition training data so no testing data is used during training
- [ ] Test Binary data
- [ ] Test data in batches
- [ ] Create confusion matrix
- [ ] Understand fix sudden drops in accuracy

## Refrences ##

[1] https://paperswithcode.com/sota/image-classification-on-mnist \
[2] https://paperswithcode.com/sota/image-classification-on-fashion-mnist
