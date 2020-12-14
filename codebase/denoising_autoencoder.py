import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image


# Load datasets

test_new_spots = np.load('test_new_spots_inpainted10k.npz',allow_pickle= True)          # original inpainted dataset
test_new_adv_spots = np.load('test_newadv_spots_inpainted10k.npz',allow_pickle= True)   # inpainted with adversarial perturbations


X = test_new_adv_spots['arr_0']
Y = test_new_spots['arr_0']

#split into train and test sets

X_train, X_test = X[:int(.8 * len(X))], X[int(.8 * len(X)):]
Y_train, Y_test = Y[:int(.8 * len(Y))], Y[int(.8 * len(Y)):]


Input_img = Input(shape=(32, 32, 3))  
    

## Network architecture

#encoder
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x2 = MaxPool2D( (2, 2))(x2)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)

# decoder
x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x3 = UpSampling2D((2, 2))(x3)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
decoded = Conv2D(3, (3, 3), padding='same')(x1)


# Initialse model 
autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1, mode='auto')

#Train model
a_e = autoencoder.fit(X_train, Y_train,
            epochs=50,
            batch_size=256,
            shuffle=True,
            validation_data=(X_test, Y_test),
            callbacks=[early_stopper])

#generate output images
imgs = []
for i in range(len(X)):
  imgs.append(autoencoder.predict(X[i].reshape(1,32,32,3)).reshape(32,32,3))