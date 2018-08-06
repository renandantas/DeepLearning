import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv("entradas-breast.csv")
classe = pd.read_csv("saidas-breast.csv")

def criarRede():
    classificador = Sequential()

    # contrução da primeira camada oculta e da camada de entrada
    # units -> 16 neuronios na primeira camada oculta
    # activation -> função de ativação
    # kernel_initializer ->
    # input_dim -> camada de entrada -> usado somente na primeira camada oculta
    classificador.add(Dense(units=16, activation='relu',
                            kernel_initializer='random_uniform', input_dim=30))
    
    # Realiza o Dropout, zera aleatoriamente X% dos neuronios da cama de entrada, evita overfitting
    classificador.add(Dropout(0.2))
    
    #criando a segunda camada oculta
    classificador.add(Dense(units=16, activation='relu',
                            kernel_initializer='random_uniform'))
    
    # Realiza o Dropout, zera aleatoriamente X% dos neuronios da cama de entrada, evita overfitting
    classificador.add(Dropout(0.2))
    
    # contrução da camada de saida
    classificador.add(Dense(units=1, activation='sigmoid'))
   
    # Configurações dos parametros de otimização
    # learn rate (lr) -> quanto menor melhor, mas gasta mais tempo para realizar os calculos
    # decay -> indica quanto o learn rate vai ser decrementado a cada atuaização de pesos
    # começa com um learn rate alto e vai diminuindo o valor do learn rate aos poucos
    # clipvalue -> vai prender o valor, os pesos ficam em uma determinada faixa de valor -> ex: 0.5 e -0.5
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    
    # metodo compile faz a configuração do modelo para o treinamento
    # Optimizer ->coloca qual a função vai ser utilizada para fazer o ajuste dos pesos (decida do gradiente)
    # Loss -> é a função de perda, onde se faz o tratamento ou o calculo do erro
    # metrics -> é a metrica que vai ser usada para a fazer a avaliação do erro
    classificador.compile(optimizer=otimizador, loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    
    return classificador

# Cria o classificador
classificador = KerasClassifier(build_fn = criarRede, epochs = 100, batch_size = 10)

# Realiza a predição com base de dados cruzada, divide as bases em treinamento e teste varias vezes
# x-> indica quais são os atributos preisores
# y -> indica quais são os resultados finais
# cv -> em quantas vezes vai ser dividida a base de dados
# scoring -> como que vai retornar a base de dados
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')

# Faz a media do acerto da base de dados
media = resultados.mean()

# faz o desvio padrao da media de acerto da base de dados, quanto maior for esse valor mais dificuldade a base de dados vai ter para reconhecer
# dados que ela nunca viu
desvio = resultados.std()