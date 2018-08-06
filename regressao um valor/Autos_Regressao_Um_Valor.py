import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv("autos.csv", encoding = "ISO-8859-1")

# Retira as colunas da base de dados que não são necessarias para as predições
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

# Vizualiza os valores de preço <= 10
i1 = base.loc[base.price <= 10]

# Apaga os dados que possuem preço menor que 10
base = base[base.price > 10]

i2 = base.loc[base.price > 350000]
base = base[base.price  < 350000]

# Realiza a substituição de valores que estão faltando pelos que mais aparecem
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}

base = base.fillna(value = valores)

# Pega as colunas que são os previsores
previsores = base.iloc[:, 1:13].values

# Pega os valores dos carros - resultado que queremos
preco_real = base.iloc[:, 0].values

# Transforma o parametro que são String em valor numerico
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

# Pré-processamento, arrumando a base de dados para uma melhor predição
# 0 0 0 0
# 2 0 1 0
# 3 0 0 1
onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10])
previsores = onehotencoder.fit_transform(previsores).toarray()

# Criando a rede neural
regressor = Sequential()
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units = 158, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear'))

# Mean absolute error considera somente o valor e desconsidera o sinal, modulo da matematica
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])

# Realiza o treinamento da rede neural
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100) 

# Realiza a predição
previsoes = regressor.predict(previsores)

# Media do preço real
preco_real.mean()

# Media das previsoes
previsoes.mean()