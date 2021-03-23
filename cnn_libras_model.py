"""
Created on Sat Mar 13 2021

@author: William M.C. Junior
"""

# Bibliotecas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense

# Definindo nossa classe
class Cnn:
    
    # método construtor da classe, que recebe como parâmetros as dimensões
    # da imagem e os rótulos(classes) 
    @staticmethod
    def build(width, heigth, depth, classes):
        
        model = Sequential()
    
        input_shape = (width, heigth, depth)
        
        # INPUT->CONV->POOL->CONV->POOL->CONV->POOL->FLATTEN->DENSE->OUTPUT
        
        model.add(Conv2D(32, (3,3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model