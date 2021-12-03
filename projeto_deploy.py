import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from sklearn import datasets


# Carregar base de dados
wine = datasets.load_wine()
df_wine = pd.DataFrame(data=wine.data,columns=wine.feature_names)

# Dividindo variáveis independentes e dependentes
df_wine['class'] = wine.target

# Dividindo dados de teste de dados de treinamento
X_train, X_test, y_train, y_test = train_test_split(df_wine.drop('class',axis=1), df_wine['class'], test_size=0.2, random_state=42)

#Definindo a função de parametrização do modelo
def print_knn_accuracy(k, metric_arg, X, y, X_test_arg, y_test_arg):
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric_arg
    )

    knn = knn.fit(X, y)

    y_pred = knn.predict(X_test_arg)

    "Accuracy:", metrics.accuracy_score(y_test_arg, y_pred).round(5) 

menu = ['Apresentação','KNN', 'DTC']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Apresentação':
    st.title('Apresentação')

    st.text('''Projeto elaborado por Gabriel Borsero e Gustavo Galisa para demonstrar estudos sobre
KNNs e DTCs, na cadeira de Tópicos Especiais do curso de TSI, no IFPB.

Serão apresentados dois modelos de aprendizagem supervisionada, ou seja, modelos que
utilizam variáveis independentes para prever uma variável dependente.

Ou seja, nos é dado um conjunto de dados rotulados que já sabemos qual é a nossa saída
correta e que deve ser semelhante ao conjunto, tendo a ideia de que existe uma relação
entre a entrada e a saída.
Problemas de aprendizagem supervisionados são classificados em problemas de “regressão”
e “classificação”.

Em um problema de regressão, estamos tentando prever os resultados em uma saída contínua,
o que significa que estamos a tentando mapear variáveis ​​de entrada para alguma função
contínua.
Exemplo: Dada uma imagem de homem/mulher, temos de prever sua idade com base em dados da
imagem.

Em um problema de classificação, estamos tentando prever os resultados em uma saída
discreta. Em outras palavras, estamos tentando mapear variáveis ​​de entrada em categorias
distintas.
Exemplo: Dada um exemplo de tumor cancerígeno, temos de prever se ele é benigno ou
maligno através do seu tamanho e idade do paciente.

Fonte: https://tinyurl.com/2p8bjpb6
''')

elif choice == 'KNN':
    st.title("KNN")

    st.text('''O algoritmo kNN é um dos algoritmos de classificação mais conhecidos e fáceis de se
implementar na literatura de aprendizado de máquina e mineração de dados. Sua ideia
consiste em, dado um objeto desconhecido, procurar pelos k vizinhos mais próximos a
ele em um conjunto de dados previamente conhecido, segundo uma medida de distância
pré-estabelecida.
    ''')

    st.text('''KNN(K — Nearest Neighbors) é um dos muitos algoritmos de aprendizagem supervisionada
usado no campo de data mining e machine learning. Ele é um classificador onde o
aprendizado é baseado 'no quão similar' é um dado (um vetor) do outro.

O algoritmo KNN assume que coisas semelhantes existem nas proximidades. Em outras
palavras, coisas semelhantes estão próximas umas das outras.

Principais Hiperparâmetros:

- N-neighbourd: número de vizinhos a serem considerados;

- Algoritmo: A opções ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’ são algoritmos
disponíveis para calcular o vizinho mais próximo;

- p: 1 ativa a utilização da distância de Manhattan e 2 da distância Euclidiana.
Caso não seja inserido nenhum input, distância de minkowski será ativada que é a
generalizaçãodas distâncias de Manhattan e Euclidiana;

- Metric: Através desse parâmetro escolhemos o tipo de distância que mais se adequa para
o problema em questão. 
''')


    st.text('''Vamos utilizar nesse exemplo o dataset Wine e o seguinte código:''')

    st.code('''
    def print_knn_accuracy(k, metric_arg, X, y, X_test_arg, y_test_arg):
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric_arg
        )

        knn = knn.fit(X, y)

        y_pred = knn.predict(X_test_arg)

        "Accuracy:", metrics.accuracy_score(y_test_arg, y_pred).round(5) 
    ''')

    st.header('Aplicação')

    st.text('''Abaixo podemos passar valores para K, definir métrica (distância) que queremos testar e
ver o resultado em tempo real, caso haja alteração:''')

    # Setta o valor de K
    k_value = st.slider('Qual o valor de K deseja atribuir?', 1, 30)

    # Setta a medida de distância
    distances_options = st.radio(
                                "Qual opção de distância gostaria de testar?",
                                ('euclidean','manhattan', 'minkowski', 'hamming'))

    # Explicações sobre as distâncias mais conhecidas
    if distances_options == 'euclidean':
        st.text('A distância euclidiana representa a distância mais curta entre dois pontos.')
    elif distances_options == 'manhattan':
        st.text('''A distância de Manhattan é a soma das diferenças absolutas entre os pontos em 
    todas as dimensões.''')
    elif distances_options == 'hamming':
        st.text('''A distância de Hamming calcula a distância entre dois vetores binários. Essa
    técnica acaba sendo uma boa opção para casos de variáveis categóricas (one-hot).''')
    elif distances_options == 'minkowski':
        st.text('''É a generalização das distâncias de Manhattan e Euclidiana.''')

    # Exibe a taxa de acerto do modelo 
    taxa_de_precisao_knn = print(print_knn_accuracy(k_value, distances_options, X_train, y_train, X_test, y_test))


    st.text('Fonte: https://tinyurl.com/3dup3kce')

elif choice == 'DTC':

    st.title('DTC')

    st.text('''Árvores de decisão são métodos de aprendizado de máquinas supervisionado não-paramétricos,
muito utilizados em tarefas de classificação e regressão, podendo ser usado para prever
categorias discretas (sim ou não, por exemplo) e para prever valores numéricos (o valor
do lucro em reais).

Assim como um fluxograma, a árvore de decisão estabelece nós (decision nodes) que se
relacionam entre si por uma hierarquia. Existe o nó-raiz (root node), que é o mais
importante, e os nós-folha (leaf nodes), que são os resultados finais. No contexto de
machine learning, o raiz é um dos atributos da base de dados e o nó-folha é a classe ou o
valor que será gerado como resposta.
''')


    st.code('''
        def print_dtc(crit):
        dtc = DecisionTreeClassifier(criterion=crit)

        dtc = dtc.fit(X_train, y_train)

        y_pred = dtc.predict(X_test)

        "Accuracy:", metrics.accuracy_score(y_test, y_pred).round(4)
    ''')

    st.header('Aplicação')
    st.text(' ')
    criterion_options = st.radio(
                        "Qual opção de criterion gostaria de testar?",
                        ('gini','entropy'))

    # Árvore de decisão
    def print_dtc_accuracy(crit, X, y, X_test_arg, y_test_arg):
        dtc = DecisionTreeClassifier(criterion=crit)
        dtc = dtc.fit(X, y)
        y_pred = dtc.predict(X_test_arg)
        
        "Accuracy:", metrics.accuracy_score(y_test_arg, y_pred).round(4)

    taxa_de_precisao_DTC = print_dtc_accuracy(criterion_options, X_train, y_train, X_test, y_test)
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text('Fonte: https://tinyurl.com/4jnwyfnh')
    st.text('Fonte: https://tinyurl.com/2p82wxde')
