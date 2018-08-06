import pandas as pd
from keras.layers import Dense, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv("games.csv")

# Pré processamento da base de dados, melhorando a base de dados
base = base.drop("Other_Sales", axis = 1)
base = base.drop("Global_Sales", axis = 1)
base = base.drop("Developer", axis = 1) 

# Deleta todas as linhas que possuem valor nan
base = base.dropna(axis = 0)

base = base.loc[base["NA_Sales"] > 1]
base = base.loc[base["EU_Sales"] > 1]

nome_jogos = base.Name
base = base.drop("Name", axis = 1) 

# Seleciona os previsores
previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values

# Resultado das vendas
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

# Transforma os dados String em dados numericos
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

# s 1 0
# r 0 1
onehotencoder = OneHotEncoder(categorical_features= [0,2,3,8])
previsores = onehotencoder.fit_transform(previsores).toarray()

# Criando a camada de entrada
camada_entrada = Input(shape = (61,))

# Criando a primeira camada oculta
camada_oculta1 = Dense(units = 32, activation = "sigmoid")(camada_entrada)

# Criando a segunda camada oculta
camada_oculta2 = Dense(units = 32, activation = "sigmoid")(camada_oculta1)

# resultado possue 3 saidas
# Criando a camada de saida
camada_saida1 = Dense(units = 1, activation = "linear")(camada_oculta2)
camada_saida2 = Dense(units = 1, activation = "linear")(camada_oculta2)
camada_saida3 = Dense(units = 1, activation = "linear")(camada_oculta2)

# Cria o regressor
regressor = Model(inputs = camada_entrada, outputs = [camada_saida1, camada_saida2, camada_saida3])

# add as configurações dos parametros da rede neural
regressor.compile(optimizer = "adam", loss = "mse")

# Reliza o treinamento
regressor.fit(previsores, [venda_na, venda_eu, venda_jp], epochs = 5000, batch_size = 100)

# Realiza a predição dos valores
previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)