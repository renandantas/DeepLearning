# Usada para visualizar as imagens que estão na base de dados 
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential

# Flatten -> Transforma uma matriz em um vetor
# Conv2D -> Camada de convolução
# MaxPooling2d -> Serve para infatizar as caracteristicas ou dos obejetos para a classificação
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Utilizado para fazer o mapeamento das variaveis dummy -> é necessario fazer uma transformação nesses dados pois temos 10 classes
from keras.utils import np_utils

# realiza a normalização na camada de convoluções
from keras.layers.normalization import BatchNormalization

# X - para atributos previsores
# y - para as classes
# Baixa as imagens para e separa o treinamento e o teste
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# Visualiza as imagens e coloca escala de cinza - tira a cor da imagem
plt.imshow(X_treinamento[1], cmap = "gray")
plt.title("Classe " + str(y_treinamento[1]))

# Transforma os dados para o TensorFlow fazer a leitura - muda o formato da imagem
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

# Variaveis precisam ser float32
previsores_treinamento = previsores_treinamento.astype("float32")
previsores_teste = previsores_teste.astype("float32")

# Mudando a escala de valores, melhora o processamento, colocando os valores em uma escala de 0 até 1
# min_max_normalization -> transforma os valores em uma escala menor que favilita o processamento
# realiza uma normalização dos dados
previsores_treinamento /= 255
previsores_teste /= 255

# Cria as variaveis dummy - é necessário pois a classificação são de mais de duas classes
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# Cria o classificador
classificador = Sequential()

# Camada 1 - Define o operador de convolução - gera uma mapa de caracteristicas
# 32 -> numero de kernels -> realiza alguns testes na imagem com "filtros"
# 3,3 -> tamanho do kernel, significa o tamanho do detector de caracteristicas - vai de acordo com o tamanho da imagem
# input_shape -> tamanho da imagem e quantidade de canais
# activation -> função de ativação
classificador.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation = "relu"))

# Realiza a normalização na camada de convolução
classificador.add(BatchNormalization())

# Camada 2 - Pooling
# pool_size -> tamanho da matriz ou da janela que vai selecionar as partes com maior valor no mapa de caracteristicas
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Camada 3 - Flattening - transforma a matriz em formato de vetor para passarmos os valores para a rede neural densa
# Quando usamos o BacthNormalization o Flattening só vai ser usado na ultima camada de convolução

#classificador.add(Flatten())

# Camada 4 - mais uma camda de concolução
classificador.add(Conv2D(32, (3,3), activation = "relu"))

# Realiza a normalização na camada de convolução
classificador.add(BatchNormalization())

# Camada 5 - Pooling
# pool_size -> tamanho da matriz ou da janela que vai selecionar as partes com maior valor no mapa de caracteristicas
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Camada 6 - Flattening - transforma a matriz em formato de vetor para passarmos os valores para a rede neural densa
classificador.add(Flatten())

# começa a criar a rede neural densa normalmente 
# Camada 7 - Primeira camada oculta
classificador.add(Dense(units = 128, activation = "relu"))

# Dropout -> Zera 20% das entradas - pois existem muitos neuronios na camada
classificador.add(Dropout(0.2))

# Camada 8 - Segunda camada oculta
classificador.add(Dense(units = 128, activation = "relu"))

# Dropout -> Zera 20% das entradas - pois existem muitos neuronios na camada
classificador.add(Dropout(0.2)) 
# Camada 5 - Camada de saÃ­da
classificador.add(Dense(units = 10, activation = "softmax"))

# Realiza a compilação da rede neural
classificador.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Realiza o treinamento
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 128, epochs = 5, validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)