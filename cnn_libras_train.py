"""
Created on Sat Mar 13 15:37:31 2021

@author: junio
"""
##############################################################################

# Meus módulos
from cnn_libras_model import Cnn
from cf_matrix import make_confusion_matrix

# Bibliotecas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from keras.models import load_model

# Auxiliares
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time

##############################################################################

# definindo funções para obter data e hora dos treinamentos dos modelos
def ObterDataFormatada():
    return str('{date:%Y%m%d_%H%M}').format(date = datetime.datetime.now())

def ObterTempoMin(inicio, fim):
    return (fim - inicio) / 60

##############################################################################

print("[INFO][INICIO] executando Script...\n" + ObterDataFormatada() + '\n')

print("[INFO] preparando e aumentando conjunto de dados...\n")

IMAGE_SHAPE = (64,64) # definindo dimensão da imagem

DATA_ROOT = 'dataset' # diretório principal
DATA_TRAINING_DIR = str(DATA_ROOT + '\\libras_training') # dir de treino/val
DATA_TEST_DIR = str(DATA_ROOT + '\\libras_test') # dir de teste

datagen_kwargs = dict(rescale=1./255, validation_split=.20)

valid_datagen = ImageDataGenerator(**datagen_kwargs)
train_datagen = ImageDataGenerator(**datagen_kwargs)
test_datagen = ImageDataGenerator(**datagen_kwargs)

# flow_from_diretory para separar os conjuntos conforme as pastas

train_generator = train_datagen.flow_from_directory(
    DATA_TRAINING_DIR,
    subset='training',
    target_size=IMAGE_SHAPE,
    shuffle=True,
    classes=['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W','Y'])

valid_generator = valid_datagen.flow_from_directory(
    DATA_TRAINING_DIR,
    subset='validation',
    target_size=IMAGE_SHAPE,
    shuffle=True,
    classes=['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W','Y'])

test_generator = test_datagen.flow_from_directory(
    DATA_TEST_DIR,
    target_size=IMAGE_SHAPE,
    shuffle=False,
    classes=['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W','Y'])

##############################################################################

# imprimindo informações de labels (classes) e tamanho de batch
print("[INFO] informações dos dados de treino...\n")

image_batch_train, label_batch_train = next(iter(train_generator))
print("image batch shape: ", image_batch_train.shape)
print("label batch shape: ", label_batch_train.shape)

dataset_labels = sorted(train_generator.class_indices.items(), 
                        key=lambda pair:pair[1])

dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)

##############################################################################

print("\n[INFO] compilando o modelo...\n")
inicio = time.time()

# chamando método construtor do modelo da classe Cnn
model = Cnn.build(64, 64, 3, train_generator.num_classes)
opt = Adam()
model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=["acc"])

print("\n[INFO] imprimindo estrutura do modelo...\n")
model.summary()

##############################################################################

# definindo número de passos por época
steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)

# definindo número de validações por época
val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)

print("\n[INFO] treinando o modelo...\n")
h = model.fit(train_generator, epochs=50, verbose=2,
              steps_per_epoch = steps_per_epoch,
              validation_data = valid_generator,
              validation_steps=val_steps_per_epoch).history
                        
##############################################################################

print("\n[INFO] salvando o modelo...\n")

FILE_NAME = 'cnn_model_libras_v1_'
file_date = ObterDataFormatada()
model.save('output/models/' + FILE_NAME + file_date + '.h5')

print("[INFO] modelo: output/models/" + FILE_NAME + file_date + '.h5 salvo')

fim = time.time()

print("[INFO] tempo de execução do modelo: %.1f min" %(ObterTempoMin(inicio, fim)))

##############################################################################

# carregando o modelo salvo para testar o conjunto de teste
model = load_model('output/models/' + FILE_NAME + file_date + '.h5')

##############################################################################

# imprimindo informações de labels (classes) e tamanho de batch
print("[INFO] informações dos dados de teste...\n")

val_image_batch, val_label_batch = next(iter(test_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)

print("validation batch shape: ", val_image_batch.shape)
print("label batch shape: ", val_label_batch.shape)

test_labels = sorted(test_generator.class_indices.items(), 
                     key=lambda pair:pair[1])

test_labels = np.array([key.title() for key, value in test_labels])

print(test_labels)

##############################################################################

print("\n[INFO] avaliando o modelo...\n")

perda_teste, acuracia_teste = model.evaluate(val_image_batch, val_label_batch)

print("Perda do teste: ", perda_teste)
print("Acurácia do teste: ", acuracia_teste)

##############################################################################

val_steps_per_epoch = np.ceil(test_generator.samples/test_generator.batch_size)
Y_pred = model.predict(test_generator, val_steps_per_epoch)
y_pred = np.argmax(Y_pred, axis=1)

print('\nConfusion Matrix\n')
print(confusion_matrix(test_generator.classes, y_pred))

print('\nQuantidade total de amostras de teste: ', test_generator.samples)

target_names = ['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W','Y']
print('\nClassification Report\n')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

##############################################################################

print("[INFO] salvando matriz de confusão...")

# chamando a função para construir a matriz de confusão formatada
cm = confusion_matrix(test_generator.classes, y_pred)
make_confusion_matrix(cm, figsize=(20,10), categories=target_names)

# salvando matriz de confusão conforme o diretório especificado
plt.savefig('output/confusion_matrix/' + FILE_NAME + file_date + '.png')

##############################################################################

print("[INFO] plottando gráficos...\n")

plt.style.use("ggplot")
plt.figure()

plt.plot(h["loss"], label="train_loss")
plt.plot(h["val_loss"], label="val_loss")
plt.plot(h["acc"], label="train_acc")
plt.plot(h["val_acc"], label="val_acc")

plt.title("Acurácia e Perca por Época")
plt.xlabel("Épocas")
plt.ylabel("Acurácia / Perca")
plt.legend()

plt.savefig('output/plots/' + FILE_NAME + file_date + '.png', bbox_inches='tight')

##############################################################################

print("[INFO] gerando imagem do modelo de camadas...\n")

plot_model(model, to_file='output/image_layers/' + FILE_NAME + file_date + '.png',
          show_shapes=True)

print("[INFO] [FIM]" + ObterDataFormatada())