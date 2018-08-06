import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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

# Cria a rede neural
def criar_Rede():
     classificador = Sequential()
     classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
     classificador.add(Dense(units = 4, activation = 'relu'))
     classificador.add(Dense(units = 3, activation = 'softmax'))
     classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
     return classificador

# Cria o classificador 
classificador = KerasClassifier(build_fn = criar_Rede, epochs = 1000, batch_size = 10)

# Realiza a classificação com base de dados cruzada
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')

# Tira a media dos acertos da rede
media = resultados.mean()

# Realiza o desvio padrão da media
desvio = resultados.std()