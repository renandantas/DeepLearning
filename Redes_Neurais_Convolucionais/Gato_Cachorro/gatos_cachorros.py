from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Reaçiza a leitura de uma imagem
from keras.preprocessing import image

# Começa a criar a rede neural
classificador = Sequential()

# Camada de convolução
# 32 -> quantidade de filtros - o ideal é começar por 64
# 3,3 -> dimensões para o detector de caracteristicas
# input_shape -> os 2 primeiros parametros é para indicar a dimensão, o ultimo parametro é o numero de canais -> 3 pois é RGB
#               -> as imagens precisam ter sempre as mesmas dimensões
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = "relu"))

# Coloca os dados entre valor de o e 1 para facilitar o processamento
classificador.add(BatchNormalization())

# Pooling - pega os valores mais altos, ou seja as caracteristicas mais relevantes
# 2,2 indica uma matriz de 2x2 que vai pegar os maiores valores
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Adicionando mais uma camada de convolução
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = "relu"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Flattening ->  transforma a matrix em um vetor  para passarmos de entrada na rede neural
classificador.add(Flatten())

# Criando a rede neural densa
# Primeira camada oculta da rede neural
classificador.add(Dense(units = 128, activation = "relu"))

# zera 20% das entradas
classificador.add(Dropout(0.2))

# segunda camada oculta da rede neural
classificador.add(Dense(units = 128, activation = "relu"))

# zera 20% das entradas
classificador.add(Dropout(0.2))

# Camada de saída
# uma saída pois temos um problema de classificação binária
classificador.add(Dense(units = 1, activation = "sigmoid"))

# Compila a rede neural
classificador.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Vai gerar as imagens que vão ser utilizadas para o trinamento
# Rescale faz a normalização dos dados da imagem
# rotation indica o grau que a imagem vai ser rotacionada
# shear faz a mudança dos pixel para outra dimensão
# height faz a faixa de mudança da altura
# zoom da zoom na imagem
gerador_treinamento = ImageDataGenerator(rescale = 1./255, rotation_range = 7, horizontal_flip = True, shear_range = 0.2, height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

# Serve para localizar a imagem
# target -> precisa ser o mesmo valor que é passado na camada de convolução
# class_mode -> é relacionado a quantidade de classes que nós temos
base_treinamento = gerador_treinamento.flow_from_directory("dataset/training_set", target_size = (64,64), batch_size = 32, class_mode = "binary")

base_teste = gerador_teste.flow_from_directory("dataset/test_set", target_size = (64,64), batch_size = 32, class_mode = "binary")

# Realiza o treinamento da rede e faz a validação na base de dados
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 / 32, epochs = 5, validation_data = base_teste, validation_steps = 1000 / 32)

# sobe a imagem na rede neural para realizar o teste
# target -> precisa ser o mesmo valor que é passado na camada de convolução
imagem_teste = image.load_img("dataset/test_set/cachorro/dog.3500.jpg", target_size = (64,64))

# Realiza a conversão da imagem
imagem_teste = image.img_to_array(imagem_teste)

# Realiza a normalização do dado a ser inserido
imagem_teste /= 255

# transforma em um formato que o TersorFlow trabalha com as imagens -> quantidade de imagens, altura, largura e o numero de canais
imagem_teste = np.expand_dims(imagem_teste, axis = 0)

# Faz a previsao da imagem
previsao = classificador.predict(imagem_teste)

# mostra a qual classe pertence o valor
base_treinamento.class_indices

