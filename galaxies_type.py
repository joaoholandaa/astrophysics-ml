import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dados = pd.read_excel('galaxies.xls', sheet_name='neargalcat')

print(dados.head())

#Quantidade de galáxias
sns.countplot(x=dados['class'])
plt.xticks(rotation=90)
plt.show()

#Renomeando classes para uma forma mais geral
def fix_classe(x):
    if(x=="IRREGULAR GALAXY"):
        return "Irregular"
    elif(x=="SPIRAL GALAXY"):
        return "Espiral"
    elif(x=="LENTICULAR GALAXY"):
        return "Lenticular"
    else:
        return x

dados['class'] = dados['class'].apply(fix_classe)

sns.countplot(x=dados['class'])
plt.xticks(rotation=90)
plt.show()

#Removendo galáxias sem classe definida
dados = dados[dados['class']!="UNIDENTIFIED"]
print(dados.head())

#Informações dos dados
print(dados.info())

#Verificando existências de NaNs
print(dados.isna().sum())

#Removendo linhas das colunas com menos elementos vazios
dados = dados.dropna(subset=['bmag','ks_mag','linear_diameter','abs_bmag'])
print(dados.isna().sum())

#Separando cada grupo de galáxias num dataframe
irregular = dados[dados['class']=='Irregular']
lenticular = dados[dados['class']=='Lenticular']
espiral = dados[dados['class']=='Espiral']

#Verificando existência de NaNs nas irregulares e substituindo pela mediana
irregular.isna().sum()
irregular['radial_velocity'] = irregular['radial_velocity'].fillna(irregular['radial_velocity'].mean())

#Verificando existência de NaNs nas lenticulares e substituindo pela mediana
lenticular.isna().sum()
lenticular['radial_velocity'] = lenticular['radial_velocity'].fillna(lenticular['radial_velocity'].mean())

#Verificando existência de NaNs nas espirais e substituindo pela mediana
espiral['radial_velocity'] = espiral['radial_velocity'].fillna(espiral['radial_velocity'].mean())

#Juntando dataframes
dados = pd.concat([espiral,lenticular,irregular])
print(dados.head())
print(dados.info())

#Analisando posição dos objetos
sns.lmplot(x='ra', y='dec', data=dados, hue='class', fit_reg=False, palette='coolwarm', size=6, aspect=2)
plt.title('Coordenadas equatoriais (RA-DEC)')
plt.xlabel('Ascenção reta (graus)')
plt.ylabel('Declinação (graus)')
plt.show()

#Removendo colunas name, ra e dec do dataframe
dados = dados.drop(['name','ra','dec'],axis=1)
print(dados.head())

#Analisando distribuição das variáveis
fig, ax = plt.subplots(2,3,figsize=(10,8))
sns.histplot(data=dados,x='bmag',hue='class',ax=ax[0][0])
sns.histplot(data=dados,x='ks_mag',hue='class',ax=ax[0][1])
sns.histplot(data=dados,x='linear_diameter',hue='class',ax=ax[0][2])

sns.histplot(data=dados,x='distance',hue='class',ax=ax[1][0])
sns.histplot(data=dados,x='radial_velocity',hue='class',ax=ax[1][1])
sns.histplot(data=dados,x='abs_bmag',hue='class',ax=ax[1][2])
plt.show()

#Convertendo variável classe para numérica
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dados['class'] = le.fit_transform(dados['class'].values)
print(dados.head())

#Determinando variáveis X e Y
X = dados.drop('class',axis=1).values
Y = dados['class'].values

#Amostras de treino e teste
from sklearn.model_selection import train_test_split
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X,Y,test_size=0.3,random_state=42)

#Métricas de classificação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
modelos = []
acuracia = []
precisao = []
recall = []
f1 = []

#Regressão Logistica
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_treino,Y_treino)
Y_pred_logreg = logreg.predict(X_teste)

acc_logreg = accuracy_score(Y_teste,Y_pred_logreg)
prec_logreg = precision_score(Y_teste,Y_pred_logreg,average='macro')
rec_logreg = recall_score(Y_teste,Y_pred_logreg,average='macro')
f1_logreg = f1_score(Y_teste,Y_pred_logreg,average='macro')

print("Regressão Logistica:")
print("Acurácia = {:0.2f}%".format(acc_logreg*100))
print("Precisão = {:0.2f}%".format(prec_logreg*100))
print("Recall = {:0.2f}%".format(rec_logreg*100))
print("F1 = {:0.2f}%".format(f1_logreg*100))

#Support Vector Machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_treino,Y_treino)
Y_pred_svc = svc.predict(X_teste)

acc_svc = accuracy_score(Y_teste,Y_pred_svc)
prec_svc = precision_score(Y_teste,Y_pred_svc,average='macro')
rec_svc = recall_score(Y_teste,Y_pred_svc,average='macro')
f1_svc = f1_score(Y_teste,Y_pred_svc,average='macro')

print("Support Vector Machine:")
print("Acurácia = {:0.2f}%".format(acc_svc*100))
print("Precisão = {:0.2f}%".format(prec_svc*100))
print("Recall = {:0.2f}%".format(rec_svc*100))
print("F1 = {:0.2f}%".format(f1_svc*100))

modelos.append("SVC")
acuracia.append(acc_svc)
precisao.append(prec_svc)
recall.append(rec_svc)
f1.append(f1_svc)

#Naive-Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_treino,Y_treino)
Y_pred_nb = nb.predict(X_teste)

acc_nb = accuracy_score(Y_teste,Y_pred_nb)
prec_nb = precision_score(Y_teste,Y_pred_nb,average='macro')
rec_nb = recall_score(Y_teste,Y_pred_nb,average='macro')
f1_nb = f1_score(Y_teste,Y_pred_nb,average='macro')

print("Naive-Bayes:")
print("Acurácia = {:0.2f}%".format(acc_nb*100))
print("Precisão = {:0.2f}%".format(prec_nb*100))
print("Recall = {:0.2f}%".format(rec_nb*100))
print("F1 = {:0.2f}%".format(f1_nb*100))

modelos.append("Naive-Bayes")
acuracia.append(acc_nb)
precisao.append(prec_nb)
recall.append(rec_nb)
f1.append(f1_nb)

#Árvore de Decisão
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_treino,Y_treino)
Y_pred_dtc = dtc.predict(X_teste)

acc_dtc = accuracy_score(Y_teste,Y_pred_dtc)
prec_dtc = precision_score(Y_teste,Y_pred_dtc,average='macro')
rec_dtc = recall_score(Y_teste,Y_pred_dtc,average='macro')
f1_dtc = f1_score(Y_teste,Y_pred_dtc,average='macro')

print("Árvore de Decisão:")
print("Acurácia = {:0.2f}%".format(acc_dtc*100))
print("Precisão = {:0.2f}%".format(prec_dtc*100))
print("Recall = {:0.2f}%".format(rec_dtc*100))
print("F1 = {:0.2f}%".format(f1_dtc*100))

modelos.append("Decision Tree")
acuracia.append(acc_dtc)
precisao.append(prec_dtc)
recall.append(rec_dtc)
f1.append(f1_dtc)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_treino,Y_treino)
Y_pred_rfc = rfc.predict(X_teste)

acc_rfc = accuracy_score(Y_teste,Y_pred_rfc)
prec_rfc = precision_score(Y_teste,Y_pred_rfc,average='macro')
rec_rfc = recall_score(Y_teste,Y_pred_rfc,average='macro')
f1_rfc = f1_score(Y_teste,Y_pred_rfc,average='macro')

print("Random Forest:")
print("Acurácia = {:0.2f}%".format(acc_rfc*100))
print("Precisão = {:0.2f}%".format(prec_rfc*100))
print("Recall = {:0.2f}%".format(rec_rfc*100))
print("F1 = {:0.2f}%".format(f1_rfc*100))

modelos.append("Random Forest")
acuracia.append(acc_rfc)
precisao.append(prec_rfc)
recall.append(rec_rfc)
f1.append(f1_rfc)

#Dataframe com os resultados
dicionario = {"Modelo" : modelos, "Acuracia" : acuracia, "Precisao" : precisao,
             "Recall" : recall, "F1" : f1}
pd_di = pd.DataFrame(dicionario)
print("Resultados:")
print(pd_di)

#Determinando o melhor modelo
pd_di = pd_di.sort_values(by='Recall',ascending=False)
print("Melhor modelo:")
print(pd_di)