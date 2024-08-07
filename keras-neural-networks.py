import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow import keras


#-------------------------------------------------------------------------------------------------------------
# Função para executar o código
def execute():
    # Imprimir as versões das bibliotecas com mensagens descritivas
    print(f"NumPy version: {np.__version__}")
    print(f"Keras version: {keras.__version__}")  # Imprime a versão do Keras

    # Criar um modelo Sequential com Input e Dense
    modelo = keras.Sequential([keras.layers.Dense(units=1,input_shape=[2],name='neuronio')])
    #modelo = keras.Sequential([ Input(shape=(2,)),   Dense(units=1, name='neuronio')    ])

    # Exibir resumo do modelo
    modelo.summary()
    
    pesos,bias = modelo.layers[0].get_weights()
    print(pesos.shape)
    print(bias)
  
  
execute()


