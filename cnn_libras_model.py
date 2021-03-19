"""
Created on Sat Mar 13 15:37:31 2021

@author: junio
"""

# Bibliotecas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K

# Definindo nossa classe
class Cnn:
    
    # método construtor da classe, que recebe como parâmetros as dimensões
    # da imagem e os rótulos(classes) 
    @staticmethod
    def build(width, heigth, depth, classes):
        
        model = Sequential()
    
        input_shape = (width, heigth, depth)
        
        # INPUT->CONV->POOL->CONV->POOL->CONV->POOL->FLATTEN->DENSE->OUTPUT
        
        model.add(Conv2D(16, (3,3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model