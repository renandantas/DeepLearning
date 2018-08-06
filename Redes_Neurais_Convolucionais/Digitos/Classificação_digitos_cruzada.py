from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

# Serve para trabalhar com a validação cruzada
from sklearn.model_selection import StratifiedKFold  

# Utilizada para mudar a semente geradora dos numeros aleatorios
seed = 5
np.random.seed(seed)

# Carrega a base de dados e faz a divisão entre classe e previsores
(X, y), (X_teste, y_teste) = mnist.load_data()

# Fazendo as transformações na base de dados  
previsores = X.reshape(X.shape[0], 28, 28, 1)
previsores = previsores.astype("float32")
previsores /= 255
classe = np_utils.to_categorical(y, 10)

# Controla a validação cruzada
# n_splits -> número de pedaços que a base de dados vai ser quebrada
# shuffle -> Pega os dados aleatoriamente
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)

# O resultado de cada execução vai entrar na lista resultados
# criação da lista vazia
resultados = []

# loop que vai dividir de forma diferente a base de dados em treinamento e teste varias vezes
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape = (classe.shape[0], 1))):
    # Criação da rede neural
    classificador = Sequential()
    
    # Camada de convolução
    classificador.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation = "relu"))
    
    # Pooling 
    classificador.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Flattening
    classificador.add(Flatten())
    
    # Camada oculta
    classificador.add(Dense(units = 128, activation = "relu"))
    
    # Camada de saída
    classificador.add(Dense(units = 10, activation = "softmax"))
    
    # Compilando a rede neural
    classificador.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    # Realiza o treinamento
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento], batch_size = 128, epochs = 5)
    
    # Indica a precisão da rede neura
    precisao = classificador.evaluate(previsores[indice_teste], classe[indice_teste])
    resultados.append(precisao[1])

# Cria a media de precisão da rede neural
media = sum(resultados) / len(resultados)