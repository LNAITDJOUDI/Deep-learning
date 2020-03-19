######## Importation des modules de keras et des bibliotheques nécessaire
from keras.optimizers import RMSprop, Adadelta, Adam
from keras import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import *
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing

sns.set()
'''
## Chargement de la base de données mnist ( est une grande base de données de chiffres manuscrits couramment 
## utilisée pour la formation t les tests dans le domaine de l'apprentissage automatique )
'''

(Xtrain, Ytrain), (
Xtest, Ytest) = mnist.load_data()  # chargement des données d'apprentissage, et test avec la fonction load_data()

print(Xtrain.shape[0])  # 60000 echantillons pour training
print(Xtest.shape[0])  # 10000 echantillons pour test
print(Ytrain.shape)  # Une variable colonne qui contient difffrente classes
print(np.unique(Ytrain))  # les classe sont [0 1 2 3 4 5 6 7 8 9]

## on regradera si les classes sont équilibrés avec un histogamme

# plot histogamme des classes
# plt.hist(Ytrain)
# plt.show()
image_index = 60
print(Ytrain[image_index])  # sa classe est 4
plt.imshow(Xtrain[image_index], cmap='Greys')
# plt.show()


# redemenssioner les données pour qu on puisse trvaillé sur keras
Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1)

## standardisation  des données des données training
Xtrain = Xtrain.astype('float32')

Xtrain /= 255

shape_input = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])

# Encodez la variable de sortie
lb = preprocessing.LabelBinarizer()
Ytrain = lb.fit_transform(Ytrain)

##Construction  du model keras
model = Sequential()
model.add(Conv2D(28, kernel_size=(4, 4), input_shape=shape_input))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
## entrainement du model construit
model.fit(x=Xtrain, y=Ytrain, epochs=3, batch_size=100)

## choisir une image des données test
image_index = 60
print(Ytest[60])  ## 7
plt.imshow(Xtest[image_index], cmap='Greys')
plt.show()
# redemenssioner les données pour qu'on puisse trvaillé sur keras
Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)

## standardisation  des données des données training et test
Xtest = Xtest.astype('float32')
Xtest /= 255

## Prediction
pred = model.predict(Xtest)

##transormation des predction
pred[pred < 0.6] = 0  # le choix du 0.6 est arbitraire
pred[pred > 0.6] = 1
pred = lb.inverse_transform(pred)
print(pred[image_index] == Ytest[image_index])

print(confusion_matrix(Ytest, pred))
print(classification_report(Ytest, pred, digits=10))
