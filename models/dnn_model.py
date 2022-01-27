# -*- coding: utf-8 -*-

"""
Created on Wed Sep 22 13:44:13 2021

@author: junio
"""

##############################################################################

# Bibliotecas

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import backend as K

##############################################################################

# Classe

class Dnn:
    
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
        
        model.add(Flatten(input_shape = input_shape))
        
        model.add(Dense(units = 80, activation = "relu"))
        model.add(Dense(units = 80, activation = "relu"))
        model.add(Dense(units = 80, activation = "relu"))
        
        model.add(Dense(classes, activation = "softmax"))
                
        #######################################################################
        
        return model

