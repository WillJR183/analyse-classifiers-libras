##############################################################################

# Bibliotecas - TF, KERAS, SKLEARN

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Bibliotecas Auxiliares

import matplotlib.pyplot as plt
import splitfolders
import numpy as np
import datetime
import random
import shutil
import time

# Módulos Internos

from models.cnn_model import Cnn
from cf_matrix import make_confusion_matrix

##############################################################################

# Funções para obter data e hora do treinamento

def ObterData():
    return str('{date:%Y%m%d_%H%M}').format(date = datetime.datetime.now())

def ObterTempoExec(inicio, fim):
    return (fim - inicio) / 60

##############################################################################

print('\n[INFO] Inicializando script...\n')

print('[INFO] Criando sub-conjuntos de dados...\n')
inicio = time.time()

# Gera a semente aleatória
random_seed = random.randint(1, 1000)

# Splitfolders para separar o conjunto de dados em sub-conjuntos

splitfolders.ratio(input = 'dataset_full', output = 'dataset_split',
                   seed = random_seed, ratio = (0.7, 0.1, 0.2),
                   group_prefix = None)

fim = time.time()
print('[INFO] Tempo de execução: %.1f min' %(ObterTempoExec(inicio, fim)))

##############################################################################

# Variáveis e Diretórios

FILE_NAME = 'cnn_model_'

DATA_TRAIN = 'dataset_split/train' # diretório de treino
DATA_VAL = 'dataset_split/val' # diretório de validação
DATA_TEST = 'dataset_split/test' # diretório de teste

SIZE = 64 # dimensão da imagem
CHANNEL = 3 # RGB
EPOCHS = 100 # quantidade de épocas
BATCH_SIZE = 64
INPUT_SHAPE = (SIZE, SIZE, CHANNEL) # formato de entrada

CLASSES = ['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S',
           'T','U','V','W','Y'] # classes do conjunto de dados

##############################################################################

print('\n[INFO] Redimensionando intensidade dos pixels...\n')

# Aplica o redimensionamento dos pixels

train_datagen = ImageDataGenerator(rescale = 1./255)
val_dataten = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

print('[INFO] Separando conjuntos e embaralhando dados...\n')

# flow_from_diretory - separa os conjuntos conforme as pastas

train_generator = train_datagen.flow_from_directory(
    directory = DATA_TRAIN,
    target_size = (SIZE, SIZE),
    classes = CLASSES,
    color_mode = 'rgb',
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = None)

val_generator = val_dataten.flow_from_directory(
    directory = DATA_VAL,
    target_size = (SIZE, SIZE),
    classes = CLASSES,
    color_mode = 'rgb',
    batch_size = BATCH_SIZE,
    shuffle=True,
    seed = None)

test_generator = test_datagen.flow_from_directory(
    directory = DATA_TEST,
    target_size = (SIZE, SIZE),
    classes = CLASSES,
    color_mode = 'rgb',
    batch_size = BATCH_SIZE,
    shuffle = False,
    seed = None)

##############################################################################

# Callbacks

early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')

callbacks_list = [early_stopping]

##############################################################################

print('[INFO] Construindo modelo neural...\n')

cnn_model = Cnn.build(width = SIZE,
                      heigth = SIZE,
                      depth = CHANNEL,
                      classes = train_generator.num_classes)

opt = Adam(learning_rate = 0.0001)

cnn_model.compile(optimizer = opt, loss = 'categorical_crossentropy',
              metrics = ['acc'])

cnn_model.summary()

# Define o número de passos por época

steps_per_epoch = np.ceil(train_generator.samples / train_generator.batch_size)

# Define o número de validações por época

val_steps_per_epoch = np.ceil(val_generator.samples / val_generator.batch_size)

print('\n[INFO] Treinando modelo...\n')
inicio = time.time()

# Treinamento

h = cnn_model.fit(train_generator, epochs = EPOCHS, verbose = 1,
              steps_per_epoch = steps_per_epoch,
              validation_data = val_generator,
              validation_steps = val_steps_per_epoch,
              callbacks=callbacks_list).history

fim = time.time()
print('\n[INFO] Duração do treinamento: %.1f min' %(ObterTempoExec(inicio, fim)))

##############################################################################

file_date = ObterData()

# Salva o modelo treinado

print('\n[INFO] Salvando o modelo...\n')

cnn_model.save('outputs/trained_models/' + FILE_NAME + file_date + '.h5')

##############################################################################

print('[INFO] Avaliando modelo com conjunto de teste...\n')

loss, acc = cnn_model.evaluate(test_generator)

print(f'\nPerda do teste: {loss:.3f}', 
      f'\nAcurácia do Teste: {acc:.2%}')

##############################################################################

# Gera a matriz de confusão clássica

test_steps_per_epoch = np.ceil(test_generator.samples / test_generator.batch_size)
Y_pred = cnn_model.predict(test_generator, test_steps_per_epoch)
y_pred = np.argmax(Y_pred, axis=1)

print('\nGerando matriz de confusão tradicional...\n')

print(confusion_matrix(test_generator.classes, y_pred))

print('\nQuantidade total de amostras de teste: ', test_generator.samples)

##############################################################################

# Gera o relatório de classificação

print('\nClassification Report\n')

print(classification_report(test_generator.classes, y_pred, target_names=CLASSES))


##############################################################################

# chama a função para construir a matriz de confusão formatada

print('[INFO] Gerando matriz de confusão formatada...\n')

cm = confusion_matrix(test_generator.classes, y_pred)
make_confusion_matrix(cm, figsize=(20,10), categories=CLASSES)

# Salva a matriz de confusão conforme o diretório especificado

plt.savefig('outputs/confusion_matrix/' + FILE_NAME + file_date + '.png', transparent=True)

##############################################################################

print('[INFO] Plottando gráficos...\n')

plt.style.use('ggplot')

# Gráfico da perda por época

plt.figure()

plt.plot(h['loss'], label = 'perda do treino')
plt.plot(h['val_loss'], label = 'perda da validação')

plt.title('Perda por Época')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.savefig('outputs/plots/' + FILE_NAME + file_date + '_loss_epoch.png')

##############################################################################

# Gráfico da acurácia por época

plt.figure()

plt.plot(h['acc'], label = 'acurácia do treino')
plt.plot(h['val_acc'], label = 'acurácia da validação')

plt.title('Acurácia por Época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.savefig('outputs/plots/' + FILE_NAME + file_date + '_acc_epoch.png')

##############################################################################

print('[INFO] Gerando imagem da arquitetura do modelo...\n')

plot_model(cnn_model, to_file='outputs/models_architecture_images/' + FILE_NAME + file_date + '.png',
          show_shapes=True)

##############################################################################

# Limpando sub-conjuntos criados

dirPath = 'dataset_split'

try:
    print('[INFO] Apagando sub-conjuntos criados...\n')
    shutil.rmtree(dirPath)
    
except OSError as e:
    print(f"Error:{ e.strerror}")
    
    
print('[INFO] [FIM]' + ObterData())

##############################################################################