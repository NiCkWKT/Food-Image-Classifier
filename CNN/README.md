# STAT4012 Project
## Food image classifier

### CNN1
- Basic CNN with Batch normalization and Max pooling
### CNN1_noBN
- Based on CNN1
- Remove batch normalization
### CNN1_dropout
- Based on CNN1
- Add dropout
### CNN1.5
- Based on CNN1
- Add augmentation
### CNN2
- Based on CNN1.5
- Add dropout

### ResNet_v1
- Make use of residual block instead of original paper implementation
- Cross validation
- Ensembling
- Lr scheduler (T_0 = 20, T_mult = 1)

### ResNet_v2
- Based on ResNet_v1
- Change Lr scheduler parameter (T_0 = 5, T_mult = 2)

### ResNet_v3
- Based on ResNet_v1
- Add two more residual block

### ResNet_v4
- Based on ResNet_v1
- Add average pooling

### Further improvements
- Scale up the ResNet
- More enhanced augmentation
  - E.g. Mixup
- Test time augmentation
