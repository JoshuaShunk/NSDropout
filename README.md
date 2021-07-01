# NSDropout

## NOTE ##
The first two files, New_Dropout.ipynb and Old_Dropout.ipynb, were used for development of initial layer and do not represent the accuracy of the most curent version.

## Error ##
   &#xfeff;   | MNIST Numbers | MNIST Fashion
------------- | ------------- | -------------
Old Dropout   | 5.59%         | 15.23%
New Dropout   | 0.06%         | 0.11%

Yes there are smaller MNIST error rates using CNNs, data augmentation, preprocessing ect with a normal dropout layer but in this test the only variables changed were the dropout layer. 
This alloud for a direct comparison between layers.

## To-Do ##

- [ ] Test Binary data
- [ ] Test data in batches
- [ ] Create confusion matrix
- [ ] Understand fix sudden drops in accuracy
