import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dados = pd.read_csv('star_type.csv')

print(dados.head())

labels = dados['Star type'].unique()

#Diagrama HR
for x in dados['Spectral Class'].unique():
    tmp = dados[dados['Spectral Class']==x]
    plt.scatter(tmp['Temperature (K)'],tmp['Absolute magnitude(Mv)'],label=x);

plt.xlim([42000,0]);
plt.ylim([25,-15]);
plt.xlabel('Temperatura (K)');
plt.ylabel('Magnitude Absoluta');
plt.legend();
plt.show();

#Contagem das estrelas por tipo
sns.countplot(dados['Star type']);
plt.xlabel('Tipo da estrela');
plt.ylabel('Contagem');
plt.show();

#Contagem das estrelas por cor
plt.figure(figsize=(8,5))
sns.countplot(dados['Star color']);
plt.xticks(rotation=60);
plt.xlabel('Cor');
plt.ylabel('Contagem');
plt.show();

#Distribuição de temperatura
sns.displot(dados['Temperature (K)'])
plt.show();

#Distribuição dos raios
sns.displot(dados['Radius(R/Ro)'])
plt.show();

#Distribuição de luminosidade
sns.displot(dados['Luminosity(L/Lo)'])
plt.show();

#Convertendo variaveis
print(dados.head())
dados = pd.get_dummies(dados, columns=["Star color", "Spectral Class"], prefix=["Star color is", "Spectral Class is"])
print(dados.head())

#Determinando variaveis
x = dados.drop(['Star type'], axis=1).values
y = dados['Star type'].values

#Amostras de teste e treino
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

#Métricas
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

#Regressão Logística
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_treino, y_treino)
y_pred_logreg = logreg.predict(x_teste)
cm_logreg = confusion_matrix(y_teste, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt='g', xticklabels = labels, yticklabels = labels)
plt.show()

acc_logreg = accuracy_score(y_teste,y_pred_logreg)
precision_logreg = precision_score(y_teste,y_pred_logreg,average='macro')
recall_logreg = recall_score(y_teste,y_pred_logreg,average='macro')
f1_logreg = f1_score(y_teste,y_pred_logreg,average='macro')

print("Logistic Regression:")
print("Acuracia = {:0.2f}%".format(acc_logreg*100))
print("Precisão = {:0.2f}%".format(precision_logreg*100))
print("Recall = {:0.2f}%".format(recall_logreg*100))
print("F1 = {:0.2f}%".format(f1_logreg*100))

#Support Vector Machine
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_treino, y_treino)
y_pred_svm = svm.predict(x_teste)
cm_svm = confusion_matrix(y_teste, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='g', xticklabels = labels, yticklabels = labels)
plt.show()

acc_svm = accuracy_score(y_teste,y_pred_svm)
precision_svm = precision_score(y_teste,y_pred_svm,average='macro')
recall_svm = recall_score(y_teste,y_pred_svm,average='macro')
f1_svm = f1_score(y_teste,y_pred_svm,average='macro')

print("Support Vector Machine:")
print("Acuracia = {:0.2f}%".format(acc_svm*100))
print("Precisão = {:0.2f}%".format(precision_svm*100))
print("Recall = {:0.2f}%".format(recall_svm*100))
print("F1 = {:0.2f}%".format(f1_svm*100))

#Árvore de decisão
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_treino, y_treino)
y_pred_dtc = dtc.predict(x_teste)
cm_dtc = confusion_matrix(y_teste, y_pred_dtc)
sns.heatmap(cm_dtc, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
plt.show()

acc_dtc = accuracy_score(y_teste,y_pred_dtc)
precision_dtc = precision_score(y_teste,y_pred_dtc,average='macro')
recall_dtc = recall_score(y_teste,y_pred_dtc,average='macro')
f1_dtc = f1_score(y_teste,y_pred_dtc,average='macro')

print("Decision Tree:")
print("Acuracia = {:0.2f}%".format(acc_dtc*100))
print("Precisão = {:0.2f}%".format(precision_dtc*100))
print("Recall = {:0.2f}%".format(recall_dtc*100))
print("F1 = {:0.2f}%".format(f1_dtc*100))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_treino, y_treino)
y_pred_rfc = rfc.predict(x_teste)
cm_rfc = confusion_matrix(y_teste, y_pred_rfc)
sns.heatmap(cm_rfc, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)

acc_rfc = accuracy_score(y_teste,y_pred_rfc)
precision_rfc = precision_score(y_teste,y_pred_rfc,average='macro')
recall_rfc = recall_score(y_teste,y_pred_rfc,average='macro')
f1_rfc = f1_score(y_teste,y_pred_rfc,average='macro')

print("Random Forest:")
print("Acuracia = {:0.2f}%".format(acc_rfc*100))
print("Precisão = {:0.2f}%".format(precision_rfc*100))
print("Recall = {:0.2f}%".format(recall_rfc*100))
print("F1 = {:0.2f}%".format(f1_rfc*100))