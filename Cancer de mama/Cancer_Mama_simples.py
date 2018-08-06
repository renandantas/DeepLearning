import pandas as pd
# Classe utilizada para a criação da rede neural
import keras
from keras.models import Sequential

# Classe para se utilizar camadas densas na rede neural
# ou seja, cada um dos neuronios é ligado com todos os neuronios da camada subsequente
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe,
                                                                                              test_size=0.25)

# Senquential() é o metodo que cria a rede neural
classificador = Sequential()

# contrução da primeira camada oculta e da camada de entrada
# units -> 16 neuronios na primeira camada oculta
# activation -> função de ativação
# kernel_initializer ->
# input_dim -> camada de entrada -> usado somente na primeira camada oculta
classificador.add(Dense(units=16, activation='relu',
                        kernel_initializer='random_uniform', input_dim=30))

#criando a segunda camada oculta
classificador.add(Dense(units=16, activation='relu',
                        kernel_initializer='random_uniform'))

# contrução da camada de saida
classificador.add(Dense(units=1, activation='sigmoid'))

# metodo compile faz a configuração do modelo para o treinamento
# Optimizer ->coloca qual a função vai ser utilizada para fazer o ajuste dos pesos (decida do gradiente)
# Loss -> é a função de perda, onde se faz o tratamento ou o calculo do erro
# metrics -> é a metrica que vai ser usada para a fazer a avaliação do erro
#classificador.compile(optimizer='adam', loss='binary_crossentropy',
#                      metrics=['binary_accuracy'])

# Configurações dos parametros de otimização
# learn rate (lr) -> quanto menor melhor, mas gasta mais tempo para realizar os calculos
# decay -> indica quanto o learn rate vai ser decrementado a cada atuaização de pesos
# começa com um learn rate alto e vai diminuindo o valor do learn rate aos poucos
# clipvalue -> vai prender o valor, os pesos ficam em uma determinada faixa de valor -> ex: 0.5 e -0.5
otimizador = keras.optimizers.Adam(lr = 0.001, decay=0.0001, clipvalue = 0.5)
classificador.compile(optimizer=otimizador, loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

# Realiza o treinamento - encontra a relação dos previsores com a classe
# batch_size ->  calcula o erro para x registro e dps faz o reajuste dos pesos
# epochs -> quantas vezes que vão ser feitas os ajuste dos pesos
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size=10, epochs=100)

# Mostra os pesos que a rede neural encontrou para as ligações da camada de entrada para a primeira camada oculta
pesos0 = classificador.layers[0].get_weights()

# Mostra os pesos que a rede neural encontrou para as ligações da primeira camada oculta para a segunda camada oculta
pesos1 = classificador.layers[1].get_weights()

# Mostra os pesos que a rede neural encontrou para as ligações da segunda camada oculta para a camada de saída
pesos2 = classificador.layers[2].get_weights()

# Passa o registro do previsores_teste para o rna e a rna vai fazer o calculo dos pesos e aplicação da função de ativivação
# e vai retornar um valor de probabilidade 
previsoes = classificador.predict(previsores_teste)

# Transforma a variavel previsoes em valores verdadeiro ou falso (sem cancer ou com cancer)
previsoes = (previsoes > 0.5)

# Compara o acerto entre a classe_teste (base de dados com resultados certos) e previsoes (base que foi gerada automaticamente pelo codigo)
# Retorna uma porcentagem de acerto -----> Utilizando sklearn
precisao = accuracy_score(classe_teste, previsoes)

# Cria uma matriz onde é possivel ter uma boa visualização em qual classe temos mais erro -----> Utilizando sklearn
# Original na horizontal e classificados automaticamente na vertical
matriz = confusion_matrix(classe_teste, previsoes)

# Retorna o valor de erro e o valor da precisao ---> utilizando o Keras
resultado = classificador.evaluate(previsores_teste, classe_teste)

