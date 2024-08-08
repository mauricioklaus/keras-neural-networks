import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow import keras
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


#-------------------------------------------------------------------------------------------------------------
# Função para executar o código
def execute():
    # Imprimir as versões das bibliotecas 
    print(f"NumPy version: {np.__version__}")
    print(f"Keras version: {keras.__version__}")  

    ### Importando os dados
    print("Importando os dados")
    iris = datasets.load_iris(return_X_y = True)
    x = iris[0]
    y = iris[1]

    ### Normalização    
    ### Os dados serão normalizados entre [0, 1], para isso utilizamos o método MinMaxScaler    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    ### Categorização
    # 
    print(y[:5])
    print(f"y.shape: {y.shape}")
    #  Converte um array de inteiros (representando classes) para um array one-hot encoding.
    y = keras.utils.to_categorical(y)
    print(f"y.shape to_categorical: {y.shape}")
    
    # Imprimindo os 5 primeiros elementos de y
    print(y[:5])
    
    datasets.load_iris()['feature_names']
    datasets.load_iris()['target_names']
    
    
    ### Criação do modelo
    # Fazemos um modelo MLP definido por 1 camada de entrada, 1 camada oculta e 1 camada de saída.
    print("Criação do modelo")
    modelo = keras.Sequential([keras.layers.InputLayer(shape=[4,],name='entrada'),
                           keras.layers.Dense(512,activation='relu',name='oculta',
                           kernel_initializer=keras.initializers.RandomNormal(seed=142)),
                           keras.layers.Dense(3,activation='softmax',name='saida')])

    # Exibir resumo do modelo
    modelo.summary()

    #sns.scatterplot(x=x[:,2],y=x[:,3],hue=y,palette='tab10')
    #plt.xlabel('comprimento (cm)',fontsize =16)
    #plt.ylabel('largura (cm)', fontsize=16)
    #plt.title('Distribuição pétalas', fontsize = 18)
    #plt.show()    
    
    #pesos,bias = modelo.layers[0].get_weights()
    #print(pesos.shape)
    #print(bias)

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, stratify = y, random_state=42)
    
    print(y_teste)
    #print(y_treino)    


    print("modelo.compile")
    modelo.compile(loss      = 'categorical_crossentropy',
                   optimizer = 'rmsprop',
                   metrics   = ['categorical_accuracy'])
  
    epocas=100
    historico = modelo.fit(x_treino, y_treino, epochs=epocas, validation_split=0.3)
    
    ### Podemos avaliar o desempenho do nosso modelo durante o treinamento com os dados de historico 
    #   através do método history e plotar o processo de aprendizado
    
    #pd.DataFrame(historico.history).plot()
    #plt.grid()
    #plt.show()
    
    #Salvar o modelo
    modelo.save('my_model.keras')

execute()


