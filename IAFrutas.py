# Inteligência Artificial Para Indentificar Frutas
# Importando Bibliotecas

import pandas as pd
from sklearn import tree

#Criando Dataframe
pomar = [[150,'lisa','maca'],
         [130,'lisa','maca'],
         [180,'irregular','laranja'],
         [160,'irregular','laranja']]

colunas = ['Peso','Superfíce','Fruta']

df = pd.DataFrame(data = pomar, columns = colunas)
df = df.replace({'lisa':1,'irregular':0,'maca':1,'laranja':0})

# Separando Classe(x) e Atributos(y)
y = df['Fruta']
x = df.drop(['Fruta'], axis = 1)

# Treinando algoritmo de aprendizado de maquina: Árvore de Decisão
clf = tree.DecisionTreeClassifier()
clf.fit(x,y)

#Interação com usuário
print("OLÁ! DESCUBRIREI QUE FRUTAS TENS!")
print(" ")
peso = float(input("Digite o peso da fruta em gramas: "))
print("  ")
print("Sobre a Superfície da Fruta:")
superficie =int( input("Para lisa >> digite 1; para irregular >> digite 0: "))
print(" ")


predicao = clf.predict([[peso,superficie]])


print("Nossa predição possui", predicao*100,"% de Predição.")
print("Concluimos que essa Fruta:")

# Condição pro Resultado
if predicao == 1:
    print("É uma maçã!")
else:
    print("É uma laranja!")


































































































































