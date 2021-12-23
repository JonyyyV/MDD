import numpy as np
import pandas as pd
import statistics as st
import math
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def main():
    path_dataset = "mtcars.csv" # Escoged bien la ruta!!
    mtcars = pd.read_csv(path_dataset) # Leemos el csv
    # Discretizamos la variable clase para convertirlo en un problema de clasificacion
    ix_consumo_alto = mtcars.mpg >= 21
    mtcars.mpg[ix_consumo_alto] = 1
    mtcars.mpg[~ix_consumo_alto] = 0
    print("Este es el dataset sin normalizar")
    print(mtcars)
    print("\n\n")
    # Ahora normalizamos los datos
    mtcars_normalizado = mtcars.loc[:, mtcars.columns != 'mpg'].apply(normalize, axis=1)
    # Añadimos la clase a nuestro dataset normalizado
    mtcars_normalizado['mpg'] = mtcars['mpg']
    print("Este es el dataset normalizado")
    print(mtcars_normalizado)
    print("\n\n")
    # Hacemos un split en train y test con un porcentaje del 0.75 Train
    ltrain,ltest=splitTrainTest(mtcars_normalizado, 0.75)
    # Separamos las labels del Test. Es como si no nos las dieran!!
    
    true_labels=ltest.pop("mpg").values.astype(int).tolist()     
    
    # Predecimos el conjunto de test
    predicted_labels=[]
    mtest=ltest.values #Transformamos el dataframe de test en una matriz de float.
    mtrain=ltrain.drop("mpg",1).values #Transformamos el dataframe de entrenamiento en una matriz de float y eliminamos la columna de clase para calcular las distancias.
    for i in range(0,len(mtest)): #Bucle que recorre cada caso de test.
        caso_test = mtest[i] 
        distancias=[]
        for j in range(0,len(mtrain)): #Bucle que recorre cada caso de train.
            caso_train=mtrain[j]
            valor=euclideanDistance2points(caso_test, caso_train) #Calculamos la distancia euclidea entre el caso de test y el de train.
            distancias.append(valor) 
            
        ordenes=np.argsort(distancias) #Ordena los casos a partir de las distancias calculadas.
        
        tresprimeros = ordenes[:3] #Cogemos los k primeros, en este caso los 3 primeros.
        
        votos=ltrain.iloc[tresprimeros,:].pop("mpg").values #Obtenemos el valor de la clase (mpg) de los k (3) casos de train más cercanos.
        #Utilizamos estos valores para votar la clase del nuevo caso de test.
        cont=[0,0]
        for w in range(0,len(votos)): #Por cada voto.
            voto=int(votos[w]) 
            cont[voto]+=1
        predicted_labels.append(cont.index(max(cont))) #La clase con más votos se guarda.
        
        
        
    # Mostramos por pantalla el Accuracy por ejemplo
    print("Accuracy conseguido:")
    print(accuracy(true_labels, predicted_labels)) 
    
    # Algun grafico? Libreria matplotlib.pyplot
    return(0)

# FUNCIONES de preprocesado
def normalize(x):
    return((x-min(x)) / (max(x) - min(x)))

def standardize(x):
    return((x-st.mean(x))/st.variance(x))

# FUNCIONES de evaluacion
def splitTrainTest(data, percentajeTrain):
    """
    Takes a pandas dataframe and a percentaje (0-1)
    Returns both train and test sets
    """
    v=np.random.rand(len(data)) #Vector con tamaño igual al del parametro data con numeros aleatorios entre 0 y 1.
    mask=v>percentajeTrain #Mascara que indica aquellos que sean mayores que el parametro percentajeTrain.
    #A partir de la mascara escogemos un set de test y de train aleatorios.
    test=data.loc[mask,:] 
    train=data.loc[~mask,:]
    return((train,test))

def kFoldCV(data, K):
    """
    Takes a pandas dataframe and the number of folds of the CV
    YOU CAN USE THE sklearn KFold function here
    How to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    return()

# FUNCION modelo prediccion
def knn(newx, data, K):
    """
    Receives two pandas dataframes. Newx consists on a single row df.
    Returns the prediction for newx
    """

    return()

def euclideanDistance2points(x,y):
    """
    Takes 2 matrix - Not pandas dataframe!
    """
    return(math.sqrt(sum((x-y)**2))) #Calcula la distancia euclidea entre un dos vectores.

# FUNCION accuracy
def accuracy(true, pred): 
    cont=0 
    for i in true:  #Por cada valor.
        if (true[i]==pred[i]): #Si los datos de ambas listas coinciden.
            cont+=1 
    return(cont/len(true)) #Se devuelve el numero de aciertos entre el numero total de casos.

if __name__ == '__main__':
    #np.random.seed(25)
    main()
