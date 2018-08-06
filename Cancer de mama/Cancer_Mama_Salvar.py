import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

# Cria a rede neural
classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# Realiza o treinamento
classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

# Coloca estrutura da rede neural em um json
classificador_json = classificador.to_json()

# Salva no disco o json com a estrutura da rede neural
with open('Classificador_Cancer_Mama.json', 'w') as json_file:
    json_file.write(classificador_json)

# Salva os pesos da rede neural     
classificador.save_weights('Classificador_Cancer_Mama.h5')