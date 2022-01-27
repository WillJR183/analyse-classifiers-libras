# -*- coding: utf-8 -*-

"""
Created on Wed Sep 22 13:44:13 2021

@author: junio
"""

##############################################################################

# Bibliotecas

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

##############################################################################

# Classe

class Cnn:
    
    # Construtor da classe: (largura, altura, canal, classes)
    @staticmethod
    def build(width, heigth, depth, classes):
        
        model = Sequential() # Inicializando modelo

        input_shape = (width, heigth, depth) # Formato de entrada de dados
        
        # Tratamento, caso o canal seja o primeiro parâmetro
        
        if K.image_data_format() == "channels_first":
            
            input_shape = (depth, width, heigth)
            
        #######################################################################
        
        # Arquitetura da rede
        
        # Primeira camada (CONV -> RELU -> POOL)
        model.add(Conv2D(8, 3, padding = "same", input_shape = input_shape))
        model.add(Activation(activation = "relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        
        # Segunda camada (CONV -> RELU -> POOL)
        model.add(Conv2D(16, 3, padding = "same"))
        model.add(Activation(activation = "relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        
        # Terceira camada (CONV -> RELU -> POOL)
        model.add(Conv2D(32, 3, padding = "same"))
        model.add(Activation(activation = "relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        
        # Camadas FC (Flatten -> Dense)
        model.add(Flatten())
        model.add(Dense(units = 16, activation = "relu"))     
        
        # Camada de saída (Softmax) para classificação
        model.add(Dense(classes, activation = "softmax"))
                
        #######################################################################
        
        return model

