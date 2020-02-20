# -*- coding: utf-8 -*-
"""
@author: SERGIO AGUILERA RAMIREZ
"""

import numpy as np
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import time


# --------------------- FUNCIONES ---------------------------------------------

##### Division train y test #####
def prepara_particiones(i,j,x,y):
    trainx = []
    trainy = []
    for k in i:
        trainx.append(x[k])
        trainy.append(y[k])
    
    testx = []
    testy = []
    for k in j:
        testx.append(x[k])
        testy.append(y[k])

    return trainx, trainy, testx, testy


## Buscar vecino mas cercano de la misma clase
def buscar_vecino_amigo_enemigo(trainx,trainy,d):
    id_amigo = id_enemigo = 0
    val_amigo = val_enemigo = 99999.0
    
    q = np.delete(trainx,d,0)
    sumatoria_x = sum(trainx[d])
    
    for j in range(len(q)):
        dis = (sumatoria_x - sum(q[j]))**2.0
        
        if trainy[j] == trainy[d] and dis < val_amigo:
            id_amigo = j
            val_amigo = dis
            
        if trainy[j] != trainy[d] and dis < val_enemigo:
            id_enemigo = j
            val_enemigo = dis
        
    return trainx[id_amigo], trainx[id_enemigo]

## Calcula el porcentaje de aciertos del clasificador
def calcula_porcentaje_acertados(trainx, trainy, testx, testy):
    
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx,trainy)
    
    cont_aciertos = 0
    for j in range(len(testx)):
        prediccion = clasificador.predict([testx[j]])
        if testy[j] == prediccion:
            cont_aciertos +=1
                
    return (100 * cont_aciertos / len(testx))


# --------------------- ALGORITMOS ---------------------------------------------
    
############## KNN ##############
def Knn(i, t, x, y):     
            
    inicio = time.time()

    ## Divido la particion en train y test
    trainx, trainy, testx, testy = prepara_particiones(i,t,x,y)  
        
    ## Llamada a la funcion para el calculo del porcentaje de acierto y tasa_clas
    tasa_clas = calcula_porcentaje_acertados(trainx,trainy,testx,testy)
    
        
    ## Array de la funcion objetivo de las particones
    funcion_obejtivo = 0.5 * tasa_clas + (1 - 0.5) * 0.0

    tiempo = time.time() - inicio
   
    return tiempo, tasa_clas, funcion_obejtivo

    
############## GREEDY ##############
def greedy(i,t,x,y):
        
        ## Control de tiempo
        inicio = time.time()
        
        ## Inicializo el vector de pesos a 0
        w = np.zeros(len(x[0]))
        
        cont_reduccion = 0
        
        ## Divido la particion en train y test
        trainx, trainy, testx, testy = prepara_particiones(i,t,x,y)   
    
        ## Buscamos los amigos y enemigos, y a su vez actualizamos el vector de pesos
        for i in range(len(trainx)):
            amigo,enemigo = buscar_vecino_amigo_enemigo(trainx,trainy,i)
            w = w + abs(trainx[i] - enemigo) - abs(trainx[i] - amigo)
        
        ## Peso de la variable mas significativa
        max_peso = max(w)
    
        ## Truncamos los valores negativos y positivos
        for t in range(len(w)):
            if w[t] < 0.0:
                w[t] = 0.0
            else:
                w[t] = w[t]/max_peso
                
            if w[t] < 0.2:
                cont_reduccion += 1
                    
        
        ## Multiplico los valores de las caracteristicas por sus pesos
        trainx *= w
        testx *= w
            
        ## Tasa de aciertos
        tasa_clas = calcula_porcentaje_acertados(trainx,trainy,testx,testy)
                    
        ## Tasa de reduccion
        tasa_red = 100 * cont_reduccion / len(w)
        
        ## Funcion objetivo
        funcion_objetivo = 0.5 * tasa_clas + (1 - 0.5) * tasa_red
        
        ## Control de tiempo
        tiempo = time.time() - inicio
    
        return tasa_red, funcion_objetivo, tasa_clas, tiempo
    
    
############## BUSQUEDA LOCAL ##############
def busqueda_local(i,t,x,y):
    ## Val inicial
    max_val = -999999.9
       
    ## Control de tiempo
    inicio = time.time()
    
    ## Array de pesos
    w = np.random.uniform(0.0,1.0,len(x[0]))
        
    ## Divido en train y test
    trainx, trainy, testx, testy = prepara_particiones(i,t,x,y)     
    
    vecinos = 0
    it = 0
    
    ## Iteramos hasta llegar al maximo de vecinos o al maximo de iteraciones
    while vecinos < 20 * len(w) and it < 15000:
        
        cont_reduccion = 0  
        
        ## Recorremos los pesos de w
        for l in range(len(w)):
            # Array de distribucion normal
            z = np.random.normal(0.0,0.3)
            it += 1
            
            w_aux = w[l]
            
            ## Cambio por mutacion normal
            w[l] += z
            if w[l] < 0.2:
                cont_reduccion +=1
                w[l] = 0.0
            if w[l] > 1.0:
                w[l] = 1.0
                      
            ## Multiplicamos los datos por sus pesos
            trainx_a = trainx * w
            testx_a = testx * w
            
            ## Calculamos el porcentaje de aciertos con el clasificador KNN
            tasa_clas = calcula_porcentaje_acertados(trainx_a,trainy,testx_a,testy)           
            
            ## Tasa de reduccion
            tasa_red = 100 * cont_reduccion / len(w)
            
            ## Valor global de la funcion objetivo
            val = 0.5 * tasa_clas + (1 - 0.5) * tasa_red
                
            # Si valor de la funcion acutal es mayor que la global actualizamos y establecemos vecinos a 0, si no
            # recuperamos el vector de pesos anterior y seguimos mutando.
            if val > max_val:
                max_val = val
                vecinos = 0
                break
            else:
                w[l] = w_aux
                vecinos += 1
                    
    # Multiplicamos los datos por sus pesos
    trainx *= w
    testx *= w
    
    ## Tasa de clasificacion
    tasa_clas = calcula_porcentaje_acertados(trainx,trainy,testx,testy)
    
    ## Tasa de reduccion
    tasa_red = 100 * cont_reduccion / len(w)
        
    ## Funcion objetivo
    funcion_objetivo = 0.5 * tasa_clas + (1 - 0.5) * tasa_red

    ## Control de tiempo
    tiempo = time.time() - inicio
             
    return tasa_red, funcion_objetivo, tasa_clas, tiempo
                  



# --------------------- MAIN ---------------------------------------------

## Fijo la semilla 
np.random.seed(2)

print("\n\n---------------------------------------------------------------------------------------------------\n")
########## COLPOSCOPY #############

fichero1, m1 = arff.loadarff('../datos/colposcopy.arff')

x = []
y = []

for i in fichero1:
    i = i.tolist()
    x.append(i[:-1])
    y.append(i[-1])

x = MinMaxScaler().fit_transform(x)

## Genero las 5 particiones
par = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
par.get_n_splits(x,y)

sum_tasa_clas1 = sum_funcion_objetivo1 = sum_tiempo1 = 0
sum_tasa_clas2 = sum_tasa_red2 = sum_funcion_objetivo2 = sum_tiempo2 = 0
sum_tasa_clas3 = sum_tasa_red3 = sum_funcion_objetivo3 = sum_tiempo3 = 0

print("## COLPOSCOPY ##")
it = 1
for i, t in par.split(x,y):
    print("\n\n**Particion ", it, "**")
    
    tiempo1, tasa_clas1, funcion_objetivo1 = Knn(i,t,x,y)     

    print("\n--Resultado KNN--")
    print("Tasa_clas: ", tasa_clas1)
    print("Funcion objetivo: ", funcion_objetivo1)
    print("Tiempo de ejecucion: ", tiempo1, "seg")

    tasa_red2, funcion_objetivo2, tasa_clas2, tiempo2 = greedy(i,t,x,y)

    print("\n--Resultado Greedy--")
    print("Tasa_red: ", tasa_red2)
    print("Tasa_clas: ", tasa_clas2)
    print("Funcion objetivo: ", funcion_objetivo2)
    print("Tiempo de ejecucion: ", tiempo2, "seg")
    
    tasa_red3, funcion_objetivo3, tasa_clas3, tiempo3 = busqueda_local(i,t,x,y)

    print("\n--Resultado BL--")
    print("Tasa_red: ", tasa_red3)
    print("Tasa_clas: ", tasa_clas3)
    print("Funcion objetivo: ", funcion_objetivo3)
    print("Tiempo de ejecucion: ", tiempo3, "seg")
   
    sum_tasa_clas1 +=tasa_clas1
    sum_funcion_objetivo1 += funcion_objetivo1
    sum_tiempo1 += tiempo1
    
    sum_tasa_clas2 +=tasa_clas2
    sum_tasa_red2 += tasa_red2
    sum_funcion_objetivo2 += funcion_objetivo2
    sum_tiempo2 += tiempo2
    
    sum_tasa_clas3 +=tasa_clas3
    sum_tasa_red3 += tasa_red3
    sum_funcion_objetivo3 += funcion_objetivo3
    sum_tiempo3 += tiempo3
    
    it +=1
    
    
print("\n\n**MEDIA**")    
print("--Resultado KNN--")
print("\nTasa clas media: ", sum_tasa_clas1/5.0)
print("Funcion objetivo media: ", sum_funcion_objetivo1/5.0 )
print("Tiempo meido: ", sum_tiempo1/5.0 )

print("\n--Resultado Greedy--")
print("\nTasa clas media: ", sum_tasa_clas2/5.0)
print("Tasa red media: ", sum_tasa_red2/5.0 )
print("Funcion objetivo media: ", sum_funcion_objetivo2/5.0)
print("Tiempo meido: ", sum_tiempo2/5.0)

print("\n--Resultado BL--")
print("\nTasa clas media: ", sum_tasa_clas3/5.0)
print("Tasa red media: ", sum_tasa_red3/5.0 )
print("Funcion objetivo media: ", sum_funcion_objetivo3/5.0)
print("Tiempo meido: ", sum_tiempo3/5.0)



print("\n\n---------------------------------------------------------------------------------------------------\n")
########## IONOSFERA #############

fichero2, m2 = arff.loadarff('../datos/ionosphere.arff')
x2 = []
y2 = []

for i in fichero2:
    i = i.tolist()
    x2.append(i[:-1])
    y2.append(i[-1])

x2 = MinMaxScaler().fit_transform(x2)

print("\n\n## IONOSFERA ##")
## Genero las 5 particiones
par = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
par.get_n_splits(x2,y2)

it = 1

sum_tasa_clas4 = sum_funcion_objetivo4 = sum_tiempo4 = 0
sum_tasa_clas5 = sum_tasa_red5 = sum_funcion_objetivo5 = sum_tiempo5 = 0
sum_tasa_clas6 = sum_tasa_red6 = sum_funcion_objetivo6 = sum_tiempo6 = 0

for i2, t2 in par.split(x2,y2):
    print("\n\n**Particion ", it, "**")
    
    tiempo4, tasa_clas4, funcion_objetivo4 = Knn(i2,t2,x2,y2)     

    print("\n--Resultado KNN--")
    print("Tasa_clas: ", tasa_clas4)
    print("Funcion objetivo: ", funcion_objetivo4)
    print("Tiempo de ejecucion: ", tiempo4, "seg")

    tasa_red5, funcion_objetivo5, tasa_clas5, tiempo5 = greedy(i2,t2,x2,y2)

    print("\n--Resultado Greedy--")
    print("Tasa_red: ", tasa_red5)
    print("Tasa_clas: ", tasa_clas5)
    print("Funcion objetivo: ", funcion_objetivo5)
    print("Tiempo de ejecucion: ", tiempo5, "seg") 
    
    tasa_red6, funcion_objetivo6, tasa_clas6, tiempo6 = busqueda_local(i2,t2,x2,y2)

    print("\n--Resultado BL--")
    print("Tasa_red: ", tasa_red6)
    print("Tasa_clas: ", tasa_clas6)
    print("Funcion objetivo: ", funcion_objetivo6)
    print("Tiempo de ejecucion: ", tiempo6, "seg")

    sum_tasa_clas4 +=tasa_clas4
    sum_funcion_objetivo4 += funcion_objetivo4
    sum_tiempo4 += tiempo4
    
    sum_tasa_clas5 +=tasa_clas5
    sum_tasa_red5 += tasa_red5
    sum_funcion_objetivo5 += funcion_objetivo5
    sum_tiempo5 += tiempo5
    
    sum_tasa_clas6 +=tasa_clas6
    sum_tasa_red6 += tasa_red6
    sum_funcion_objetivo6 += funcion_objetivo6
    sum_tiempo6 += tiempo6
    
    it+=1


print("\n\n**MEDIA**")    
print("\n--Resultado KNN--")
print("\nTasa clas media: ", sum_tasa_clas4/5.0)
print("Funcion objetivo media: ", sum_funcion_objetivo4/5.0 )
print("Tiempo meido: ", sum_tiempo4/5.0 )

print("\n--Resultado Greedy--")
print("\nTasa clas media: ", sum_tasa_clas5/5.0)
print("Tasa red media: ", sum_tasa_red5/5.0 )
print("Funcion objetivo media: ", sum_funcion_objetivo5/5.0)
print("Tiempo meido: ", sum_tiempo5/5.0)

print("\n--Resultado BL--")
print("\nTasa clas media: ", sum_tasa_clas6/5.0)
print("Tasa red media: ", sum_tasa_red6/5.0 )
print("Funcion objetivo media: ", sum_funcion_objetivo6/5.0)
print("Tiempo meido: ", sum_tiempo6/5.0)



print("\n\n---------------------------------------------------------------------------------------------------\n")
########## TEXTURE #############
fichero3, m3 = arff.loadarff('../datos/texture.arff')

x3 = []
y3 = []

for i in fichero3:
    i = i.tolist()
    x3.append(i[:-1])
    y3.append(i[-1])

x3 = MinMaxScaler().fit_transform(x3)


## Genero las 5 particiones
par = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
par.get_n_splits(x3,y3)

sum_tasa_clas7 = sum_funcion_objetivo7 = sum_tiempo7 = 0
sum_tasa_clas8 = sum_tasa_red8 = sum_funcion_objetivo8 = sum_tiempo8 = 0
sum_tasa_clas9 = sum_tasa_red9 = sum_funcion_objetivo9 = sum_tiempo9 = 0

print("\n\n## TEXTURE ##")
   
it = 1      
for i3, t3 in par.split(x3,y3):  
    print("\n\n**Particion ", it, "**")
      
    tiempo7, tasa_clas7, funcion_objetivo7 = Knn(i3,t3, x3, y3)     

    print("\n--Resultado KNN--")
    print("Tasa_clas: ", tasa_clas7)
    print("Funcion objetivo: ", funcion_objetivo7)
    print("Tiempo de ejecucion: ", tiempo7, "seg")
    
    tasa_red8, funcion_objetivo8, tasa_clas8, tiempo8 = greedy(i3,t3,x3,y3)

    print("\n\n--Resultado Greedy--")
    print("Tasa_red: ", tasa_red8)
    print("Tasa_clas: ", tasa_clas8)
    print("Funcion objetivo: ", funcion_objetivo8)
    print("Tiempo de ejecucion: ", tiempo8, "seg")
    
    tasa_red9, funcion_objetivo9, tasa_clas9, tiempo9 = busqueda_local(i3,t3,x3,y3)

    print("\n--Resultado BL--")
    print("Tasa_red: ", tasa_red9)
    print("Tasa_clas: ", tasa_clas9)
    print("Funcion objetivo: ", funcion_objetivo9)
    print("Tiempo de ejecucion: ", tiempo9, "seg")
    
    sum_tasa_clas7 +=tasa_clas7
    sum_funcion_objetivo7 += funcion_objetivo7
    sum_tiempo7 += tiempo7
    
    sum_tasa_clas8 +=tasa_clas8
    sum_tasa_red8 += tasa_red8
    sum_funcion_objetivo8 += funcion_objetivo8
    sum_tiempo8 += tiempo8
    
    sum_tasa_clas9 +=tasa_clas9
    sum_tasa_red9 += tasa_red9
    sum_funcion_objetivo9 += funcion_objetivo9
    sum_tiempo9 += tiempo9
    
    it +=1
    
print("\n\n**MEDIA**")    
print("\n--Resultado KNN--")
print("\nTasa clas media: ", sum_tasa_clas7/5.0)
print("Funcion objetivo media: ", sum_funcion_objetivo7/5.0 )
print("Tiempo meido: ", sum_tiempo7/5.0 )

print("\n--Resultado Greedy--")
print("\nTasa clas media: ", sum_tasa_clas8/5.0)
print("Tasa red media: ", sum_tasa_red8/5.0 )
print("Funcion objetivo media: ", sum_funcion_objetivo8/5.0)
print("Tiempo meido: ", sum_tiempo8/5.0)

print("\n--Resultado BL--")
print("\nTasa clas media: ", sum_tasa_clas9/5.0)
print("Tasa red media: ", sum_tasa_red9/5.0 )
print("Funcion objetivo media: ", sum_funcion_objetivo9/5.0)
print("Tiempo meido: ", sum_tiempo9/5.0)