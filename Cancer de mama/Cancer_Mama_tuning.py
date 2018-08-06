import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

# Cria a rede neural
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return classificador

# Realiza o funcionamento da rede neural    
classificador = KerasClassifier(build_fn = criarRede)

# Lista de paramentros a ser usada para ver qual dos parametros tem melhor performance e resultado
parametros = {'batch_size': [10, 30], 'epochs': [50, 100], 
              'optimizer': ['adam', 'sgd'], 'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'], 'activation': ['relu', 'tanh'], 
              'neurons': [16, 8 ]}

# Executa a rede para cada parametro
grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, scoring = 'accuracy', cv = 5)

# Realiza o treinamento da rede neural - encontra a relação dos previsores com a classe
grid_search = grid_search.fit(previsores, classe)

# Encontra os melhores paramentros para se usar com essa base de dados
melhores_parametros = grid_search.best_params_

# Retorna o valor com a melhor precisao que foi achada
melhor_precisao = grid_search.best_score_