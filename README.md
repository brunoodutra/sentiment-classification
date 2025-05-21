# Repositório de Análise de Sentimentos com Dataset
=====================================================

## Introdução
---------------

Este repositório apresenta um exemplo de uso de dataset para análise de sentimentos, utilizando técnicas de Exploratory Data Analysis (EDA) e Machine Learning (ML) para classificar textos como positivos ou negativos. Além disso, é realizado um comparativo com o uso de uma Linguagem de Modelo de Grande Escala (LLM) como o LLaMA-3.3-70B.

## Objetivos
--------------

* Realizar uma análise exploratória dos dados (EDA) para entender a distribuição e características do dataset.
* Preprocessar os dados para melhorar a qualidade e relevância para a análise de sentimentos.
* Treinar e testar modelos de ML (Random Forest, SVM e Regressão Logística) para classificar textos como positivos ou negativos.
* Utilizar uma LLM (Llama-3.3-70B) para classificar textos e comparar os resultados com os modelos de ML.
* Avaliar o custo vs desempenho das abordagens utilizadas.

## Dataset
------------

* O dataset utilizado é um conjunto de textos rotulados como positivos ou negativos.

## Análise Exploratória (EDA)
---------------------------

* Foram realizadas as seguintes etapas de EDA:
 + Análise de dados: verificação de distribuição, presença de outliers e relação entre as variáveis.
 + Remoção de colunas irrelevantes: remoção de colunas que não contribuem para a análise de sentimentos.
 + Verificação de dados nulos: identificação de dados nulos e substituição por valores adequados.
 + Análise de distribuição: visualização da distribuição dos dados para entender a natureza dos dados.

## Preprocessamento de Dados
-----------------------------

* Foram realizadas as seguintes etapas de preprocessamento:
 + Ajuste da distribuição: normalização ou padronização dos dados para melhorar a estabilidade dos modelos.
 + Limpeza de pontuação: remoção de pontuação e símbolos especiais dos textos.
 + Padronização em caixa baixa: conversão de todos os textos para caixa baixa.
 + Vetorização de textos: transformação dos textos em vetores numéricos para uso nos modelos de ML.

## Modelos de Machine Learning
------------------------------

* Foram utilizados os seguintes modelos de ML:
 + Random Forest: modelo de ensemble que combina várias árvores de decisão para classificar os textos.
 + SVM: modelo de classificação que utiliza um espaçamento de vetores para separar as classes.
 + Regressão Logística: modelo de regressão que utiliza uma função logística para prever a probabilidade de uma classe.
* Foram realizadas as seguintes etapas de treinamento e teste:
 + Divisão em teste e treino: divisão dos dados em conjuntos de treinamento e teste.
 + Treinamento dos modelos: treinamento dos modelos com os dados de treinamento.
 + Teste dos modelos: teste dos modelos com os dados de teste.
 + Avaliação dos resultados: avaliação dos resultados utilizando métricas de desempenho como precisão, recall e F1-score.

## Utilização de LLM (Llama-3.3-70B)
---------------------------------

* Foram realizadas as seguintes etapas de utilização da LLM:
 + Configuração da LLM: configuração da LLM para classificar textos como positivos ou negativos.
 + Treinamento da LLM: treinamento da LLM com os dados de treinamento.
 + Teste da LLM: teste da LLM com os dados de teste.
 + Avaliação dos resultados: avaliação dos resultados utilizando métricas de desempenho como precisão, recall e F1-score.

## Comparativo de Resultados
---------------------------

* Foram realizadas as seguintes etapas de comparativo de resultados:
 + Comparação dos resultados dos modelos de ML e da LLM.
 + Avaliação do custo vs desempenho das abordagens utilizadas.

## Conclusões
--------------

* O repositório apresenta um exemplo de uso de dataset para análise de sentimentos, utilizando técnicas de EDA e ML para classificar textos como positivos ou negativos.
* A LLM (Llama-3.3-70B) apresentou resultados competitivos em relação aos modelos de ML, especialmente em termos de precisão e recall.
* A escolha da abordagem depende do custo e do desempenho necessário para a tarefa específica.

## Requisitos
-------------

* Python 3.x
* Bibliotecas necessárias: pandas, numpy, scikit-learn, transformers, etc.

## Execução
-------------

* Clonar o repositório: `git clone https://github.com/username/repository.git`
* Instalar as bibliotecas necessárias: `pip install -r requirements.txt`
* Executar o script de EDA: `python eda.py`
* Executar o script de treinamento e teste dos modelos de ML: `python train.py`
* Executar o script de utilização da LLM: `python llama.py`
* Visualizar os resultados: `python results.py`