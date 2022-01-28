## Repositório - Trabalho de Conclusão de Curso (TCC)

Este repositório contém arquivos referentes ao meu TCC , que aborda uma análise comparativa entre as técnicas de Redes Neurais Densas Comuns (DNN) , Redes Neurais Convolucionais (CNN) e Florestas Aleatórias (RF) na tarefa de classificação de imagens de gestos estáticos de Libras.

#### Diretórios :file_folder:

* *dataset_full* : contém o conjunto de dados utilizado para treino / validação e teste dos modelos.
* *models* : contém as arquiteturas dos modelos.
* *outputs* : contém os arquivos gerados durante a fase de treinamento e otimização dos modelos.
* *written_work* : contém o trabalho escrito.
* *temp* : contém o crop gerado em tempo de execução pelo script [CNN_real_time_classifier.py](https://github.com/WillJR183/analyse-classifiers-libras/blob/master/CNN_real_time_classifier.py).

#### Arquivos :page_facing_up:

* [CNN_train.py](https://github.com/WillJR183/analyse-classifiers-libras/blob/master/CNN_train.py) : script de treinamento do modelo da CNN.
* [DNN_train.py](https://github.com/WillJR183/analyse-classifiers-libras/blob/master/DNN_train.py) : script de treinamento do modelo da DNN.
* [C_RF_Train.ipynb](https://github.com/WillJR183/analyse-classifiers-libras/blob/master/C_RF_Train.ipynb) : script de treinamento do modelo do RF.
* [CNN_real_time_classifier.py](https://github.com/WillJR183/analyse-classifiers-libras/blob/master/CNN_real_time_classifier.py) : script de reconhecimento em tempo real utilizando o modelo treinado da CNN.

#### Requisitos :bookmark_tabs:

Para executar os scripts de treinamento dos modelos é necessário possuir as dependências, para isso basta instalar um gerenciador de dependências e ambientes, como o [Conda](https://docs.conda.io/en/latest/conda.html) e executar o código abaixo para importar o ambiente com as dependências utilizadas no projeto.

* Importar : ```conda env create -f environment.yml```

* Ativar : ```conda activate machinelearning```

* Execute os scripts normalmente em uma IDE Python ou via terminal : ```python CNN_train.py ```

  

#### Referências :link:

* Conjunto de dados disponível neste [repositório](https://github.com/lucaaslb/cnn-libras) do autor [Lucas Lacerda](https://github.com/lucaaslb).
