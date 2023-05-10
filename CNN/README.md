# STAT4012 Project: Food Image Classifier

**Convolutional Neural Network**
### CNN1
- Basic CNN with Batch normalization and Max pooling

### CNN1_noBN
- Based on CNN1
- Remove batch normalization

### CNN1_dropout
- Based on CNN1
- Add dropout layer

### CNN1.5
- Based on CNN1
- Add augmentation

### CNN2 (Best basic CNN model)
- Based on CNN1.5
- Add dropout layer

### ResNet_v1
- Make use of residual block instead of original paper implementation
- Cross validation
- Ensembling
- Lr scheduler (T_0 = 20, T_mult = 1)

### ResNet_v2
- Based on ResNet_v1
- Change Lr scheduler parameter (T_0 = 5, T_mult = 2)

### ResNet_v3 (Best performance)
- Based on ResNet_v1
- Add two more residual blocks

### ResNet_v4
- Based on ResNet_v1
- Add average pooling
