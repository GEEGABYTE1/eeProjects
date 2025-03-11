# EEG Movement Classification Model

CNN to classify hand movements (right, left, or circular) based on EEG. 

## Dataset Preprocessing
The raw EEG Data: 128 Hz, 14 channels --> 128 x 14 matrix per reading. 

A fast Fourier Transform was used to extract 0-30 Hz frequency components. This resulted in 30x14 matrix. However, for feature extraction, weighted & arithmetic mean was applied to each wave type. This resulted in a final matrix of 14x4x2 matrix (14 channels, 4 wave classifications, 2 mean types). 

## CNN Architecture
Input layer accepted 112 EEG features per sample. The convolutional layers extracted the temporal EEG patterns, and the ReLU layers ensured non-linearity. Dropout layers were used to prevent overfitting by randomly dropping connections. 

### Attempt for Model Optimization
The loss function used: Categorical Crossentropy. 

Batch size of 32 was used with 50 epochs. A dropout of 0.3 was used to reduce overfitting. 

## Model Performance
CNN (3 layer): 81.19% 
XGBoost: 83% 

The model is trained on 3 datasets located: `Dataset/user_a.csv`, `Dataset/user_b.csv`, `Dataset/user_c.csv`. 

## Running the Model

Ensure you have the requried libraries installed:
```
pip install tensorflow numpy pandas scikit-learn xgboost
```

Run `rnn2.ipynb` to train the model:
```
!jupyter notebook rnn2.ipynb
```