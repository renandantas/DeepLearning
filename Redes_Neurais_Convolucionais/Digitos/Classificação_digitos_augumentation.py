from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

# Gera novas imagens para o treinamento
from keras.preprocessing.image import ImageDataGenerator
# Baixa a base de dados e divide em treinamento e em teste
# x -> previsores
# y -> classe
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# Transforma os dados para o TensorFlow fazer a leitura - muda o formato da imagem
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

# Transforma os previsores em um valor float32
previsores_treinamento = previsores_treinamento.astype("float32")
previsores_teste = previsores_teste.astype("float32")

# divide os dados por 255 para o valor dos dados ficar entre 0 e 1 para um melhor processamento
previsores_treinamento /= 255
previsores_teste /= 255

# Cria as variaveis dummy
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# começa a criar a rede neural
classificador = Sequential()

# Camada de convolução
classificador.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation = "relu"))

# Pooling
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
classificador.add(Flatten())

# Primeira camada oculta
classificador.add(Dense(units = 128, activation = "relu"))

# Camada de saida
classificador.add(Dense(units = 10, activation = "softmax"))

# compilação da rede neural
classificador.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# vai gerar as novas imagens para a base de dados de treinamento e de teste
# rotation_range -> grau da rotação da imagem
# horizontal_flip -> fazer giros horizontais
# shear_range -> faz alterações no valor dos pixels -> muda os pixels de direção
# height_shift_range -> faz a alteração na faixa de altura da imagem
# zoom_range -> muda o zoom da imagem
gerador_treinamento = ImageDataGenerator(rotation_range = 7, horizontal_flip = True, shear_range = 0.2, height_shift_range = 0.07, zoom_range = 0.2)
gerador_teste = ImageDataGenerator()

# Cria a base de dados de treinamento
base_treinamento = gerador_treinamento.flow(previsores_treinamento, classe_treinamento, batch_size = 128)

# Cria a base de dados de teste
base_teste = gerador_teste.flow(previsores_treinamento, classe_treinamento, batch_size = 128)

# realiza o treinamento
# steps_per_epoch -> numero de etapas de lotes de amostra de imagem a serem geradas pelo gerador antes de declarar uma epoca concluida -> quantidade de imagens dividido pelo batch_size
classificador.fit_generator(base_treinamento, steps_per_epoch = 60000 / 128, epochs = 5, validation_data = base_teste, 
                            validation_steps = 10000 / 128)


