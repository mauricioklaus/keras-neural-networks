import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow import keras
import tensorflow as tf

#-------------------------------------------------------------------------------------------------------------
# Função para executar o código
def execute():
    # Imprimir as versões das bibliotecas 
    print(f"NumPy version: {np.__version__}")
    print(f"Keras version: {keras.__version__}")  

    
    # Carregar o modelo salvo
    modelo_carregado = keras.models.load_model('my_model.keras')
    
    # Convertendo a lista para um Tensorflow Tensor
    entrada = tf.constant([[0.61, 0.5, 0.69, 0.79]]) 
    
    # Obter as previsões
    previsoes = modelo_carregado.predict(entrada)

    # Imprimir as previsões com detalhes
    print(f"Previsões: {previsoes}")
    print(f"Tipo das previsões: {type(previsoes)}")
    print(f"Forma das previsões: {previsoes.shape}")
    print(f"Previsões como lista: {previsoes.tolist()}")

    # Encontrar o índice da classe com a maior probabilidade
    indice_da_classe_mais_provavel = np.argmax(previsoes)
    print(f"Índice da classe mais provável: {indice_da_classe_mais_provavel}")

    # Obter a probabilidade da classe mais provável
    probabilidade_da_classe_mais_provavel = previsoes[0][indice_da_classe_mais_provavel]
    print(f"Probabilidade da classe mais provável: {probabilidade_da_classe_mais_provavel}")
    
execute()


