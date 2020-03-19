from  skmultilearn.problem_transform import LabelPowerset,BinaryRelevance,ClassifierChain
from skmultilearn.ensemble import RakelD,RakelO
from skmultilearn.adapt import MLkNN
from skmultilearn.dataset import load_dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold



## Chargement des données emotions

Xtrain,Ytrain, nom_variable,nom_label=load_dataset('emotions','train') # données train
Xtest,Ytest,_,_=load_dataset('emotions','test') # données test
## description des données sur le line suivant : http://scikit.ml/tutorial.html
#Xtrain,Xtest,Ytrain,Ytest sont chargé comme matrice spase de scipy
''' On pourrais travaillé directement avec des matrices sparse avec sklearn et skmultilearn'''


## Tansformation des matrice sparse pour la visualisation des données
datatrain=pd.DataFrame(Xtrain.toarray(),columns=nom_variable)
print(datatrain.shape) # (391,72)
#print(datatrain.head(5))
#print(datatrain.info())
#print(datatrain.describe())
datatrainy=pd.DataFrame(Ytrain.toarray(),columns=nom_label)
print(datatrainy.shape)  #6 sorties donc 6 label non exculsives


## prétraitement des données

print(np.sum(datatrain.isna().sum(axis=1))) # count the number of NaN (0)
vare=VarianceThreshold(threshold=(0.001))



## Création des model multi label
## Methode 1 : BinaryRelevance (

print(Xtrain.shape)
clf = BinaryRelevance(
    classifier=RandomForestClassifier(max_depth=200),
    require_dense=[False, True]
)
anova_clf = Pipeline([('anova', vare), ('binary', clf)])
anova_clf.fit(Xtrain,Ytrain)
pred=anova_clf.predict(Xtest)
matrix=multilabel_confusion_matrix(Ytest,pred)
accuracy=accuracy_score(Ytest,pred)
print(accuracy)

## Methode 2 : LabelPowerset


clf = LabelPowerset(
    classifier=RandomForestClassifier(max_depth=200),
    require_dense=[False, True]
)
anova_clf = Pipeline([('anova', vare), ('label', clf)])
anova_clf.fit(Xtrain,Ytrain)
pred=anova_clf.predict(Xtest)
matrix=multilabel_confusion_matrix(Ytest,pred)
accuracy=accuracy_score(Ytest,pred)
print(accuracy)

#Methode 3 : Chaineclassifieur


clf =clf = ClassifierChain(
    classifier=RandomForestClassifier(max_depth=200),
    require_dense=[False, True])
anova_clf = Pipeline([('anova', vare), ('chaine', clf)])
anova_clf.fit(Xtrain,Ytrain)
pred=anova_clf.predict(Xtest)
matrix=multilabel_confusion_matrix(Ytest,pred)
accuracy=accuracy_score(Ytest,pred)
print(accuracy)

#Methode  4 : onevsrest
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
anova_clf = Pipeline([('anova', vare), ('oneVSrest', clf)])
anova_clf.fit(Xtrain,Ytrain)
pred=anova_clf.predict(Xtest)
matrix=multilabel_confusion_matrix(Ytest,pred)
accuracy=accuracy_score(Ytest,pred)
print(accuracy)

## Methode 5 : Rakel

clf =clf = RakelD(labelset_size=2,base_classifier=RandomForestClassifier())
anova_clf = Pipeline([('anova', vare), ('Rekel', clf)])
anova_clf.fit(Xtrain,Ytrain)
pred=anova_clf.predict(Xtest)
matrix=multilabel_confusion_matrix(Ytest,pred)
accuracy=accuracy_score(Ytest,pred)
print(accuracy)










