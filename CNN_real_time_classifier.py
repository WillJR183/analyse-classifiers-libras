##############################################################################

# Bibliotecas

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

##############################################################################

# Carregando modelo treinado

classificador = load_model('outputs/trained_models/cnn_model_20211012_1756.h5')

# Tamanho da imagem de captura 

img_x, img_y = 64, 64

# Quantidade de classes e dicionário de gestos e rótulos

CLASSES = 21
LETRAS = {'0' : 'A', '1' : 'B', '2' : 'C' , '3': 'D', '4': 'E', '5':'F', '6':'G',
          '7': 'I', '8':'L', '9':'M', '10':'N', '11': 'O', '12':'P', '13':'Q',
          '14':'R', '15':'S', '16':'T', '17':'U', '18':'V', '19':'W','20':'Y'}

##############################################################################

# Função preditora, lê a imagem da pasta temp, transforma em array numpy
# Realiza a predição e retorna a classe com maior probabilidade

def preditor():
    imagem_capturada = image.load_img('temp/img.png', target_size=(64,64))
    imagem_capturada = image.img_to_array(imagem_capturada)
    imagem_capturada = np.expand_dims(imagem_capturada, 0)
    resultado = classificador.predict(imagem_capturada)
    
    maior, class_index = -1, -1
    
    for x in range(CLASSES):
        if(resultado[0][x] > maior):
            maior = resultado[0][x]
            class_index = x
            
    return [resultado, LETRAS[str(class_index)]]

##############################################################################

# Inicializa a webcam nativa

cam = cv2.VideoCapture(0)

img_counter = 0
img_text = ['','']

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    img = cv2.rectangle(frame, (425,100), (625,300), (0, 0, 255), 3, 8, 0)
    crop_img = img[102:298, 427:623]
    
    cv2.putText(frame, str(img_text[1]), (30,400), cv2.FONT_HERSHEY_TRIPLEX, 
                3.0, (0, 255, 0))
    cv2.imshow('CNN - Classificacao em tempo real', frame)
        
    img_path = 'temp/img.png'
    save_img = cv2.resize(crop_img, (img_x, img_y))
    cv2.imwrite(img_path, save_img)
    img_text = preditor()
    print(str(img_text[0]))
    
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

##############################################################################