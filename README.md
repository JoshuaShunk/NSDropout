# NSDropout (Neuron Specific Dropout)

## NOTE ##
The first two files, New_Dropout.ipynb and Old_Dropout.ipynb, were used for the development of the initial layer and do not represent the accuracy of the most current version.

## Error rate (%) ##
Error rate recored as 1 - accuracy.
|   &#xfeff;             | MNIST Numbers | MNIST Fashion  | CIFAR-10
|-------------           | ------------- | -------------  | ---------
|Old Dropout             | 5.59          | 15.23          |  0
|New Dropout             | 0.00*         | 0.19           |  7.72
|Highest Reported[1][2]  | 0.13          | 3.09           |  0

Yes, there are smaller MNIST error rates using CNNs, data augmentation, preprocessing, etc with a normal dropout layer but in this test the only variables changed were the dropout layer. This allowed for a direct comparison between layers and the improvements the new layer made.

 _*No images were miss-classified. The model trained on 9,600 images and validated on the full 10000. Testing images were split up 8000 for optimization and 1600 for the new dropout layer. See mnist_numbers_implementation_of_New_Dropout.ipynb for more information._

## Testing Methodology ##

With saving the model from the best epoch not set up I let every model run for 1000 epochs (both old dropout models' validation accuracy was decreasing at this point) and recorded the epoch where its validation accuracy was the highest. Thanks to setting the NumPy seed I was able to run the training again and stop at the epoch where the model previously hit its high.

## Development ##

The goal of this layer was to take an input from the previous layer, the true values, the output of the validation data at the same location, and the true value of the validation data. Then sort the data into classes using the true values of training and validation. Next average the data of the classes and subtract the validation data from the training data. After that, the model creates a mask setting the highest X differences equal to 0 and the rest to 1 for each class. Finally, apply the appropriate mask to each of the input values bases on their true mask. 

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

During testing, I noticed that validation or testing accuracy was never going up but training was. As one of my solutions to this problem, I decided to add an inference layer that takes the last used mask(eventually when saving the best model is implemented the mask that goes with that model) and applies it to the output of the previous layer when testing. The solution can be found below.
```python
 def infrence(self, input, label):
     self.input = input
     self.label = label
     idx = np.argsort(self.label)
     input_sorted = input[idx]
     label_sorted = label[idx]
     self.infrence_binary_mask = np.empty(shape=self.input.shape)
     for i, (input, label) in enumerate(zip(self.input, self.label)):
       for true, diff in enumerate(self.diff_mask):
         if label == true:
           self.infrence_binary_mask[i] = self.diff_mask[diff]

     self.output = self.infrence_binary_mask * self.input
```
## Deployment (current state) ##

Layer needs to be initialized by calling:
```python
dropout1 = Layer_CatagoricalNSDropout(rate)
```
Rate will be replaced by the percent of neurons you want to be turned off and 1 replaced by the number of dropout layers you have. Inside the training loop, the following needs to be called:
```python
dropout1.forward(X=activation1.output, y=y_train, X_test=cached_val_inputs, y_test=y_test)
```
Activation1.output will be replaced by the output of the layer right before, y_trian should be the true training values, cached_val_inputs will be the output of the same layer as where the dropout layer is placed. For example, if the new dropout layer is placed after activation1, X_test will be the output of activation1 when running a validation pass. Finally, y_test is the true values of the testing data. Unlike classical dropout, the new dropout layer requires a presence when inferencing. To do this call:
```python
dropout1.infrence(activation1.output,y_test)
```
## Future Testing ##

I'm currently working on testing CIFAR10, but more importantly, I'm working on collecting more data and smoothing out data for better comparison between the current dropout layer and the new dropout layer as well as gain a deeper understanding of why the new layer makes prolonged drops inaccuracy. Once those issues are sorted out, I hope to migrate the layers into a more Keras/PyTorch-like model to make creating new models magnitudes easier. This includes adding a feature such as model.add(layer), model.predict(data), and training in batches. At that point, I will dive into where this layer improves over the existing layer and where it falls short, and if a variation could be used in NLP or active machine vision. 

## To-Do ##

- [X] Partition training data so no testing data is used during training
- [ ] Test Binary data
- [ ] Test data in batches
- [X] Create confusion matrix
- [ ] Understand fix sudden drops in accuracy
- [ ] Test multiple layers

## Refrences ##

[1] https://paperswithcode.com/sota/image-classification-on-mnist \
[2] https://paperswithcode.com/sota/image-classification-on-fashion-mnist
