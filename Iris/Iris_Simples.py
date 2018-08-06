import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix

# importa o csv
base = pd.read_csv("iris.csv")

# Realiza a divisão da base de dados em previsores e classe
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Transforma os valores da classe que estão em String para numeros
labelenconder = LabelEncoder()
classe = labelenconder.fit_transform(classe)

# Transforma a classe em um atributo categorico
classe_dummy = np_utils.to_categorical(classe)

# iris setosa       1 0 0
# iris virginica    0 1 0
# iris versicolor   0 0 1

# Divide a base em 25% para teste e o restante para treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

# Criando a rede neural
classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)

# Retorna o valor da função de perda e o valor da precisão
resultado = classificador.evaluate(previsores_teste, classe_teste)

# Retorna uma probabilidade para cada dado
previsoes = classificador.predict(previsores_teste)

# Transforma em verdadeiro ou falso
previsores = (previsoes > 0.5)

# Retorna o indice da coluna com o resultado final - usado para fazer a matriz de confusão nesse caso
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in classe_teste]

# Matriz de confusão
matriz = confusion_matrix(previsoes2, classe_teste2)
