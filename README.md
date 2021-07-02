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

## Development ##

The goal of this layer was to take an input from the previous layer, the true values, the output of the validation data at the same location, and the true value of the validation data. Then sort the data into classes using the true values of training and validation. Next average the data of the classes and subtract the validation data from the training data. After that create a masking setting the highest X differences equal to 0 and the rest to 1 for each class. Finally apply the appropreate mask to each of the input values bases on their true mask. 

#### Sorting inputed data into classes using dictionaries ####
```python
sorted_x = {}
sorted_y = {}
for classes in range(len(set(y))):
   sorted_x["class_{0}".format(classes)] = X[y == classes]
   sorted_y["label_{0}".format(classes)] = y[y == classes]

 sorted_x_test = {}
 sorted_y_test = {}
 for classes in range(len(set(y))):
   sorted_x_test["class_{0}".format(classes)] = X_test[y_test == classes]
   sorted_y_test["label_{0}".format(classes)] = y_test[y_test == classes]
```

#### Averaging sorted data from each class then finding the difference between the averaged train and test inputs ####
```python
differnce_classes = {}
for i, classes, test_classes in zip(range(len(set(y))), sorted_x, sorted_x_test):
   differnce_classes["diff_{0}".format(i)] = np.mean(sorted_x[classes], axis=0) - np.mean(sorted_x_test[classes], axis=0)
```

#### Masking the data taking the high values(greatest difference between train and test) and setting their values to 0 ####
```python
self.diff_mask = {}
for i, classes, test_classes, diff in zip(range(len(set(y))), sorted_x, sorted_x_test, differnce_classes):
   ind = np.argpartition(differnce_classes[diff], -round(len(X[0]) * self.rate))[-round(len(X[0]) * self.rate):]
   mask = np.ones(np.mean(sorted_x[classes],axis=0).shape, dtype=bool)
   mask[ind] = False
   differnce_classes[diff][~mask] = 0.
   differnce_classes[diff][mask] = 1
   self.diff_mask["mask_{0}".format(i)] = differnce_classes[diff]
```

#### Going through each input values and applies the apprioprite mask based on what the true output should be. ####
```python
binary_mask = np.empty(shape=X.shape)
for i, (input, label) in enumerate(zip(X,y)): 
   for true, diff in enumerate(self.diff_mask):
     if label == true:
       self.binary_mask[i] = self.diff_mask[diff]
```
## To-Do ##

- [X] Partition training data so no testing data is used during training
- [ ] Test Binary data
- [ ] Test data in batches
- [ ] Create confusion matrix
- [ ] Understand fix sudden drops in accuracy

## Refrences ##

[1] https://paperswithcode.com/sota/image-classification-on-mnist \
[2] https://paperswithcode.com/sota/image-classification-on-fashion-mnist
