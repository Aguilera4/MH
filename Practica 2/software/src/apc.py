# -*- coding: utf-8 -*-
"""
@author: SERGIO AGUILERA RAMIREZ
"""

import numpy as np
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import random
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
     
    trainy = np.array(trainy)
    
    
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


# --------------------- ALGORITMOS ---------------------------------------------
    
###########################       KNN       ###########################
def Knn(trainx, trainy, testx, testy):     
    inicio = time.time()
    
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx,trainy)
    
    prediccion = clasificador.predict(testx)
    
    tasa_clas = metrics.accuracy_score(testy,prediccion) * 100
       
    ## Array de la funcion objetivo de las particones
    funcion_objetivo = 0.5 * tasa_clas
    
    tiempo = time.time() - inicio
   
    return tiempo, tasa_clas, funcion_objetivo

    

###########################       RELIEF       ###########################
def greedy(trainx, trainy, testx, testy):
        
        ## Control de tiempo
        inicio = time.time()
        
        ## Inicializo el vector de pesos a 0
        w = np.zeros(len(trainx[0]))
    
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
                
        # Contamos el numero de reduccion
        cont_reduccion = np.count_nonzero(w<0.2)
        w[w<0.2] = 0.0        

        
        ## Tasa de reduccion
        tasa_red = 100 * cont_reduccion / len(w)
        
        ## Multiplico los valores de las caracteristicas por sus pesos
        trainx *= w
        testx *= w
         
        # Clasificador 1NN
        clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        clasificador.fit(trainx,trainy)
        prediccion = clasificador.predict(testx)
    
        # Calculo el porcentaje de aciertos
        tasa_clas = metrics.accuracy_score(testy,prediccion) * 100
                    
        ## Funcion objetivo
        funcion_objetivo = 0.5 * (tasa_clas + tasa_red)

        ## Control de tiempo
        tiempo = time.time() - inicio
    
        return tasa_red, funcion_objetivo, tasa_clas, tiempo
    


# FUNCIÓN OBJETIVO
def calcula_funcion_objetivo(ind_near,trainy,cont_reduccion,w):    
 
    tasa_clas = np.mean(trainy[ind_near] == trainy) * 100
    
    ## Tasa de reduccion
    tasa_red = 100 * cont_reduccion / len(w)          
    ## Valor global de la funcion objetivo
    funcion_objetivo = 0.5 * tasa_clas + 0.5 * tasa_red
    
    return funcion_objetivo, tasa_clas, tasa_red



###########################       BL       ###########################
def busqueda_local(trainx, trainy, testx, testy):       
    ## Control de tiempo
    inicio = time.time()
    
    ## Array de pesos
    w = np.random.uniform(0.0,1.0,len(trainx[0]))
    
    vecinos = 0
    it = 0
    
    cont_reduccion = np.count_nonzero(w<0.2)
    
    w_trunc = w.copy()
    w_trunc[w_trunc<0.2] = 0.0    
    
    trainx_a = (trainx * w_trunc)
    
    # Clasificador 1NN
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)
    
    ind_near = clasificador.kneighbors(trainx_a,n_neighbors=2)[1][:,1]    

    funcion_objetivo, tasa_clas, tasa_red = calcula_funcion_objetivo(ind_near, trainy, cont_reduccion, w)
    
    ## Iteramos hasta llegar al maximo de vecinos o al maximo de iteraciones
    while vecinos < 20 * len(trainx[0]) and  it < 15000:
        
        cont_reduccion = 0  
        
        ## Recorremos los pesos de w
        for l in range(len(w)):
            # Array de distribucion normal
            z = np.random.normal(0.0,0.3)
            it += 1
            
            w_aux = w[l]
            
            ## Cambio por mutacion normal
            w[l] += z

            if w[l] < 0.0:
                w[l] = 0.0
            if w[l] > 1.0:
                w[l] = 1.0
                       
            w_trunc = w.copy()
            
            cont_reduccion = np.count_nonzero(w_trunc<0.2)
            w_trunc[w_trunc<0.2] = 0.0
 
            trainx_a = (trainx * w_trunc)

            # Clasificador 1NN
            clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
            clasificador.fit(trainx_a,trainy)
    
            ind_near = clasificador.kneighbors(trainx_a, n_neighbors=2)[1][:,1] 
            
            val, tasa_clas, tasa_red = calcula_funcion_objetivo(ind_near, trainy, cont_reduccion, w_trunc)
            
            # Si valor de la funcion acutal es mayor que la global actualizamos y establecemos vecinos a 0, si no
            # recuperamos el vector de pesos anterior y seguimos mutando.
            if val > funcion_objetivo:
                funcion_objetivo = val
                vecinos = 0
                break
            else:
                w[l] = w_aux
                vecinos += 1
                
    
    cont_reduccion = np.count_nonzero(w<0.2)
    w[w<0.2] = 0.0
                
    trainx_a = (trainx * w)  
    testx_a = (testx * w)
    
     # Clasificador 1NN
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)
    prediccion = clasificador.predict(testx_a)
    
    
    # Calculo el porcentaje de aciertos
    tasa_clas = metrics.accuracy_score(testy,prediccion) * 100
    tasa_red = cont_reduccion / len(w) * 100
                    
    ## Funcion objetivo
    funcion_objetivo = (tasa_clas + tasa_red) * 0.5

    ## Control de tiempo
    tiempo = time.time() - inicio
             
    return tasa_red, funcion_objetivo, tasa_clas, tiempo
                  



# GENERACIÓN DE POBLACIÓN INICIAL
def genera_poblacion(x,num_individuos):
    poblacion = []
    for i in range(num_individuos):
        poblacion.append(np.random.uniform(0,1,len(x[0])))
    
    return poblacion


# EVALUACIÓN DE POBLACIÓN
def evalua_poblacion(trainx,trainy,poblacion, it):
    eval_poblacion = []
    for i in range((len(poblacion))):
        cont_reduccion = np.count_nonzero(poblacion[i]<0.2)
        poblacion[i][poblacion[i]<0.2] = 0.0
        
        trainx_a = trainx * poblacion[i]
            
        # Clasificador 1NN
        clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        clasificador.fit(trainx_a,trainy)
    
        ind_near = clasificador.kneighbors(trainx_a,n_neighbors=2)[1][:,1]    
        it += 1
        tasa_clas = np.mean(trainy[ind_near] == trainy) * 100
        tasa_red = cont_reduccion /len(poblacion[i]) * 100
        funcion_objetivo = (tasa_clas + tasa_red) * 0.5
        
        eval_poblacion.append(funcion_objetivo)
        
    return eval_poblacion, it


# SELECCIÓN
def torneo_binario(poblacion,eval_poblacion): 
    p1 = np.random.randint(0,len(poblacion))
    p2 = np.random.randint(0,len(poblacion))

    while(p1 == p2):
        p2 = np.random.randint(0,len(poblacion))

    indi1 = poblacion[p1]
    indi2 = poblacion[p2]
    
    if eval_poblacion[p1] >= eval_poblacion[p2]:
        return indi1
    else:
        return indi2


# CRUCE BLX
def cruce_BLX(cromosoma1, cromosoma2):    
    hijos = []
    
    hijo1 = np.zeros(len(cromosoma1))
    hijo2 = np.zeros(len(cromosoma1))
    
    for j in range(len(cromosoma1)):
        cmax = np.max([cromosoma1[j],cromosoma2[j]])
        cmin = np.min([cromosoma1[j],cromosoma2[j]])
        I = cmax - cmin
        
        hijo1[j] = np.random.uniform(cmin-I*0.3,cmax+I*0.3)
        hijo2[j] = np.random.uniform(cmin-I*0.3,cmax+I*0.3)    
    
    hijo1[hijo1<0.0] = 0.0
    hijo1[hijo1>1.0] = 1.0
    
    hijo2[hijo2<0.0] = 0.0
    hijo2[hijo2>1.0] = 1.0
            
    hijos.append(hijo1)
    hijos.append(hijo2)
          
    return hijos

        
# CRUCE ARITMÉTICO
def cruce_aritmetico(cromosoma1, cromosoma2): 
    hijos = []
           
    hijo1 = 0.3 * cromosoma1 + 0.7 * cromosoma2
    hijo2 = 0.7 * cromosoma1 + 0.3 * cromosoma2
    
    hijo1[hijo1<0.0] = 0.0
    hijo1[hijo1>1.0] = 1.0
    
    hijo2[hijo2<0.0] = 0.0
    hijo2[hijo2>1.0] = 1.0 
    
    hijo1 = np.array(hijo1, np.float64)
    hijo2 = np.array(hijo2, np.float64)
    
    hijos.append(hijo1)
    hijos.append(hijo2)
    
    return hijos



###########################       AGG       ###########################
def AGG(trainx, trainy, testx, testy,tipo_cruce):
    ## Control de tiempo
    inicio = time.time()
    
    # Probabilidad de mutadción
    prob_mu = 0.001
    it = 0
    
    # Genero la poblacion inicial y la evaluo
    poblacion = genera_poblacion(trainx,30)
    eval_poblacion, it = evalua_poblacion(trainx,trainy,poblacion, it)
    
    while it < 15000:
        
        # Selección de padres
        padres = []
        for i in range(len(poblacion)):
            padres.append(torneo_binario(poblacion,eval_poblacion))
                       
        hijos = []
        num_cruces = 0
        max_cruces = round(0.7 * len(padres)/2)
        
        # Cruces
        if tipo_cruce == 'BLX':
            while num_cruces < max_cruces:
                res_cruce = cruce_BLX(padres[num_cruces],padres[num_cruces+1])
                hijos.append(res_cruce[0])
                hijos.append(res_cruce[1])
                num_cruces += 2
        else:
            while num_cruces < max_cruces:
                res_cruce = cruce_aritmetico(padres[num_cruces],padres[num_cruces+1])                
                hijos.append(res_cruce[0])
                hijos.append(res_cruce[1])
                num_cruces += 2
            
                
        # Añado los padres que no han sido cruzados
        for i in range(max_cruces,30):
            hijos.append(padres[i])          
            
        # Mutaciones
        num_genes = len(hijos[0]) * len(hijos)
        num_esperado_mutaciones = round(prob_mu * num_genes)
        mut = 0
        
        while mut < num_esperado_mutaciones:
            i = np.random.randint(0,len(hijos))
            j = np.random.randint(0,len(hijos[0]))
            
            z = np.random.normal(0.0,0.3)
            hijos[i][j] += z
            
            if hijos[i][j] > 1.0:
                hijos[i][j] = 1.0
            if hijos[i][j] < 0.0:
                hijos[i][j] = 0.0
            mut += 1
        
        eval_hijos, it = evalua_poblacion(trainx,trainy, hijos, it)
                
        # Elitismo
        id_max = eval_poblacion.index(max(eval_poblacion))
        id_min = eval_hijos.index(min(eval_hijos))
        
        hijos.pop(id_min)
        eval_hijos.pop(id_min)
        hijos.append(poblacion[id_max])
        eval_hijos.append(eval_poblacion[id_max])
           
        # Actualizo población y evaluo
        poblacion = hijos.copy()
        eval_poblacion = eval_hijos.copy()
        
    
    # Selecciono el individuo con mayor valor de funcion objetivo
    id_max = eval_poblacion.index(max(eval_poblacion))
    
    cont_reduccion = np.count_nonzero(poblacion[id_max]<0.2)
    poblacion[id_max][poblacion[id_max]<0.2] = 0.0
 
    trainx_a = trainx * poblacion[id_max]
    testx_a = testx * poblacion[id_max]
        
     # Clasificador 1NN
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)
    prediccion = clasificador.predict(testx_a)
    
    # Calculo el porcentaje de aciertos
    tasa_clas = metrics.accuracy_score(testy,prediccion) * 100
    tasa_red = cont_reduccion / len(poblacion[id_max]) * 100
                    
    ## Funcion objetivo
    funcion_objetivo = (tasa_clas + tasa_red) * 0.5
    
    ## Control de tiempo
    tiempo = time.time() - inicio
             
    return tasa_red, funcion_objetivo, tasa_clas, tiempo




###########################       AGE       ###########################
def AGE(trainx, trainy, testx, testy,tipo_cruce):
    ## Control de tiempo
    inicio = time.time()
    
    # Probabilidad de mutadción
    prob_mu = 0.001
    
    it = 0
    
    # Genero la poblacion inicial y la evaluo
    poblacion = genera_poblacion(trainx,30)
    eval_poblacion, it = evalua_poblacion(trainx,trainy,poblacion, it)
    
    while it < 15000:
        # Selección de dos padres
        padres = []
        for i in range(2):
            padres.append(torneo_binario(poblacion,eval_poblacion))
                  
        # Cruces  
        hijos = []
        if tipo_cruce == 'BLX':              
                res_cruce = cruce_BLX(padres[0],padres[1])
                hijos.append(res_cruce[0])
                hijos.append(res_cruce[1])             
        else:
                res_cruce = cruce_aritmetico(padres[0],padres[1])   
                hijos.append(res_cruce[0])
                hijos.append(res_cruce[1])

        # Mutaciones     
        mutaciones = []        
        pm_cromosoma = prob_mu * len(hijos[0])       
        for i in range(2):
            n = random.random()
            if n < pm_cromosoma:
                j = np.random.randint(0,len(hijos[0]))
                z = np.random.normal(0.0,0.3)
                hijos[i][j] += z
        
                if hijos[i][j] > 1.0:
                    hijos[i][j] = 1.0
                if hijos[i][j] < 0.0:
                    hijos[i][j] = 0.0

        mutaciones = hijos.copy()
        
        # Evaluo las mutaciones
        eval_mutaciones, it = evalua_poblacion(trainx,trainy,mutaciones, it)
        
        # Composición de la nueva población (competición de las mutaciones)
        for i in range(2):
            id_min = eval_poblacion.index(min(eval_poblacion))
            if eval_poblacion[id_min] < eval_mutaciones[i]:
                poblacion.pop(id_min)
                eval_poblacion.pop(id_min)
                poblacion.append(mutaciones[i])
                eval_poblacion.append(eval_mutaciones[i])

    # Selecciono el individuo con mayor valor de funcion objetivo
    id_max = eval_poblacion.index(max(eval_poblacion))
    
    cont_reduccion = np.count_nonzero(poblacion[id_max]<0.2)
    poblacion[id_max][poblacion[id_max]<0.2] = 0.0
 
    trainx_a = trainx * poblacion[id_max]
    testx_a = testx * poblacion[id_max]
        
     # Clasificador 1NN
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)
    prediccion = clasificador.predict(testx_a)
    
    # Calculo el porcentaje de aciertos
    tasa_clas = metrics.accuracy_score(testy,prediccion) * 100
    tasa_red = cont_reduccion / len(poblacion[id_max]) * 100
                    
    ## Funcion objetivo
    funcion_objetivo = (tasa_clas + tasa_red) * 0.5
    
    ## Control de tiempo
    tiempo = time.time() - inicio
             
    return tasa_red, funcion_objetivo, tasa_clas, tiempo




###########################       AM (cada 10 generaciones)      ###########################
def AM(trainx, trainy, testx, testy, tipo_probabilidad):
    ## Control de tiempo
    inicio = time.time()
    
    # Probabilidad de mutadción
    prob_mu = 0.001
    
    it = 0
    generacion = 1
    
    # Genero la poblacion inicial y la evaluo
    poblacion = genera_poblacion(trainx,10)
    eval_poblacion, it = evalua_poblacion(trainx,trainy,poblacion, it)
    
    
    while it < 15000:
        
        # Selección de padres
        padres = []
        for i in range(len(poblacion)):            
            padres.append(torneo_binario(poblacion,eval_poblacion))
            
        # Cruces con BLX
        hijos = []
        num_cruces = 0
        max_cruces = round(0.7 * len(padres)/2)
        while num_cruces < max_cruces:
            res_cruce = cruce_BLX(padres[num_cruces],padres[num_cruces+1])
            hijos.append(res_cruce[0])
            hijos.append(res_cruce[1])
            num_cruces += 2
         
        # Añado los padres que no han sido cruzados
        for i in range(max_cruces,10):
            hijos.append(padres[i])
             
        
       # Mutaciones     
        mutaciones = []          
        num_genes = len(hijos[0]) * len(hijos)
        num_esperado_mutaciones = round(prob_mu * num_genes)
        mut = 0
        
        if num_esperado_mutaciones == 0:
           num_esperado_mutaciones = 1

        while mut < num_esperado_mutaciones:
            i = np.random.randint(0,len(hijos))
            j = np.random.randint(0,len(hijos[0]))
            
            z = np.random.normal(0.0,0.3)
            hijos[i][j] += z
            
            if hijos[i][j] > 1.0:
                hijos[i][j] = 1.0
            if hijos[i][j] < 0.0:
                hijos[i][j] = 0.0
            mut += 1
            
        mutaciones = hijos.copy()
        
        eval_mutaciones , it = evalua_poblacion(trainx,trainy,mutaciones, it)     
        
        # Elitismo
        id_max = eval_poblacion.index(max(eval_poblacion))
        id_min = eval_mutaciones.index(min(eval_mutaciones))
        
        mutaciones.pop(id_min)
        eval_mutaciones.pop(id_min)
        mutaciones.append(poblacion[id_max])
        eval_mutaciones.append(eval_poblacion[id_max])
        
        
        # Aplico Busqueda Local            
        pobla_ls = []
        eval_ls = []
        if generacion % 10 == 0 and generacion != 0:
            if(tipo_probabilidad == 0):
                for i in range(len(mutaciones)):
                    agr, w, it = busqueda_local_geneticos(trainx, trainy, testx, testy, mutaciones[i], it)
                    pobla_ls.append(w)
                    eval_ls.append(agr) 
                mutaciones = pobla_ls.copy()
                eval_mutaciones = eval_ls.copy()
                
            elif  tipo_probabilidad == 1:
                num_busquedas = round(0.1 * len(mutaciones))
                for i in range(num_busquedas):
                    j = np.random.randint(0,len(mutaciones))
                    agr, w, it = busqueda_local_geneticos(trainx, trainy, testx, testy, mutaciones[j], it)
                    mutaciones.pop(j)
                    eval_mutaciones.pop(j)
                    mutaciones.append(w) 
                    eval_mutaciones.append(agr)
                    
            elif tipo_probabilidad == 2 :
                num_busquedas = round(0.1 * len(mutaciones))
                for i in range(num_busquedas):
                    j = eval_mutaciones.index(max(eval_mutaciones))
                    agr, w, it = busqueda_local_geneticos(trainx, trainy, testx, testy, mutaciones[j], it)
                    mutaciones.pop(j)
                    eval_mutaciones.pop(j)
                    mutaciones.append(w) 
                    eval_mutaciones.append(agr)
          
        # Actualizo la población y la evaluo
        poblacion = mutaciones.copy()
        eval_poblacion = eval_mutaciones.copy()
        generacion += 1
        
        
    # Selecciono el individuo con mayor valor de funcion objetivo
    id_max = eval_poblacion.index(max(eval_poblacion))
    
    cont_reduccion = np.count_nonzero(poblacion[id_max]<0.2)
    poblacion[id_max][poblacion[id_max]<0.2] = 0.0
 
    trainx_a = trainx * poblacion[id_max]
    testx_a = testx * poblacion[id_max]
        
     # Clasificador 1NN
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)
    prediccion = clasificador.predict(testx_a)
    
    # Calculo el porcentaje de aciertos
    tasa_clas = metrics.accuracy_score(testy,prediccion) * 100
    tasa_red = cont_reduccion / len(poblacion[id_max]) * 100
                    
    ## Funcion objetivo
    funcion_objetivo = (tasa_clas + tasa_red) * 0.5
    
    ## Control de tiempo
    tiempo = time.time() - inicio
             
    return tasa_red, funcion_objetivo, tasa_clas, tiempo


###########################       BUSQUEDA LOCAL GENÉTICOS       ###########################
def busqueda_local_geneticos(trainx, trainy, testx, testy, w, it1):           
    vecinos = 0
    
    cont_reduccion = np.count_nonzero(w<0.2)
    
    w_trunc = w.copy()
    w_trunc[w_trunc<0.2] = 0.0    
    
    trainx_a = (trainx * w_trunc)
    
    # Clasificador 1NN
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)
    
    ind_near = clasificador.kneighbors(trainx_a,n_neighbors=2)[1][:,1]    
    it1+=1
    funcion_objetivo, tasa_clas, tasa_red = calcula_funcion_objetivo(ind_near, trainy, cont_reduccion, w)
    
    ## Iteramos hasta llegar al maximo de vecinos o al maximo de iteraciones
    while vecinos < 2*len(w):
        
        cont_reduccion = 0  
        
        ## Recorremos los pesos de w
        for l in range(len(w)):
            # Array de distribucion normal
            z = np.random.normal(0.0,0.3)
            
            w_aux = w[l]
            
            ## Cambio por mutacion normal
            w[l] += z

            if w[l] < 0.0:
                w[l] = 0.0
            if w[l] > 1.0:
                w[l] = 1.0
                       
            w_trunc = w.copy()
            
            cont_reduccion = np.count_nonzero(w_trunc<0.2)
            w_trunc[w_trunc<0.2] = 0.0
 
            trainx_a = (trainx * w_trunc)

            # Clasificador 1NN
            clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
            clasificador.fit(trainx_a,trainy)
    
            ind_near = clasificador.kneighbors(trainx_a, n_neighbors=2)[1][:,1] 
            it1+=1
            val, tasa_clas, tasa_red = calcula_funcion_objetivo(ind_near, trainy, cont_reduccion, w_trunc)
            
            # Si valor de la funcion acutal es mayor que la global actualizamos y establecemos vecinos a 0, si no
            # recuperamos el vector de pesos anterior y seguimos mutando.
            if val > funcion_objetivo:
                funcion_objetivo = val
                vecinos = 0
                break
            else:
                w[l] = w_aux
                vecinos += 1
                             
    return funcion_objetivo, w, it1



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

print("## COLPOSCOPY ##")
it = 1

agr_10 = agr_01 = agr_01m = 0
for i, t in par.split(x,y):
    print("\n\n**Particion ", it, "**")
    
    ## Divido la particion en train y test
    trainx, trainy, testx, testy = prepara_particiones(i,t,x,y)  
    
    
    tiempo1, tasa_clas1, funcion_objetivo1 = Knn(trainx, trainy, testx, testy)     

    print("\n--Resultado KNN--")
    print("Tasa_clas: ", tasa_clas1)
    print("Funcion objetivo: ", funcion_objetivo1)
    print("Tiempo de ejecucion: ", tiempo1, "seg")
    
    
    tasa_red2, funcion_objetivo2, tasa_clas2, tiempo2 = greedy(trainx, trainy, testx, testy)

    print("\n--Resultado Greedy--")
    print("Tasa_red: ", tasa_red2)
    print("Tasa_clas: ", tasa_clas2)
    print("Funcion objetivo: ", funcion_objetivo2)
    print("Tiempo de ejecucion: ", tiempo2, "seg")
    
    
    tasa_red3, funcion_objetivo3, tasa_clas3, tiempo3 = busqueda_local(trainx, trainy, testx, testy)

    print("\n--Resultado BL--")
    print("Tasa_red: ", tasa_red3)
    print("Tasa_clas: ", tasa_clas3)
    print("Funcion objetivo: ", funcion_objetivo3)
    print("Tiempo de ejecucion: ", tiempo3, "seg")
    
    
    ## En el AGG y AGE se modifica la ultima entrada para alternar entre BLX y CA
    tasa_redAGG1, funcion_objetivoAGG1, tasa_clasAGG1, tiempoAGG1 = AGG(trainx, trainy, testx, testy,'BLX')
    
    print("\n--Resultado AGG--")
    print("Tasa_red: ", tasa_redAGG1)
    print("Tasa_clas: ", tasa_clasAGG1)
    print("Funcion objetivo: ", funcion_objetivoAGG1)
    print("Tiempo de ejecucion: ", tiempoAGG1, "seg")
    
    
    
    tasa_redAGE, funcion_objetivoAGE, tasa_clasAGE, tiempoAGE = AGE(trainx, trainy, testx, testy,'BLX')
    
    print("\n--Resultado AGE--")
    print("Tasa_red: ", tasa_redAGE)
    print("Tasa_clas: ", tasa_clasAGE)
    print("Funcion objetivo: ", funcion_objetivoAGE)
    print("Tiempo de ejecucion: ", tiempoAGE, "seg")
    
    
    
    
    tasa_redAM_1_0, funcion_objetivoAM_1_0, tasa_clasAM_1_0, tiempoAM_1_0 = AM(trainx, trainy, testx, testy, 0)
    
    print("\n--Resultado AM_1.0--")
    print("Tasa_red: ", tasa_redAM_1_0)
    print("Tasa_clas: ", tasa_clasAM_1_0)
    print("Funcion objetivo: ", funcion_objetivoAM_1_0)
    print("Tiempo de ejecucion: ", tiempoAM_1_0, "seg")
    
    
    tasa_redAM_0_1, funcion_objetivoAM_0_1, tasa_clasAM_0_1, tiempoAM_0_1 = AM(trainx, trainy, testx, testy,1)
    
    print("\n--Resultado AM_0.1--")
    print("Tasa_red: ", tasa_redAM_0_1)
    print("Tasa_clas: ", tasa_clasAM_0_1)
    print("Funcion objetivo: ", funcion_objetivoAM_0_1)
    print("Tiempo de ejecucion: ", tiempoAM_0_1, "seg")
    
    
    tasa_redAM_0_1_mejores, funcion_objetivoAM_0_1_mejores, tasa_clasAM_0_1_mejores, tiempoAM_0_1_mejores = AM(trainx, trainy, testx, testy,2)
    
    print("\n--Resultado AM_0.1_mejores--")
    print("Tasa_red: ", tasa_redAM_0_1_mejores)
    print("Tasa_clas: ", tasa_clasAM_0_1_mejores)
    print("Funcion objetivo: ", funcion_objetivoAM_0_1_mejores)
    print("Tiempo de ejecucion: ", tiempoAM_0_1_mejores, "seg")
       
    it +=1



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

agr_102 = agr_012 = agr_01m2 = 0
for i2, t2 in par.split(x2,y2):
    print("\n\n**Particion ", it, "**")
    
    ## Divido la particion en train y test
    trainx, trainy, testx, testy = prepara_particiones(i2,t2,x2,y2) 
    
    tiempo4, tasa_clas4, funcion_objetivo4 = Knn(trainx, trainy, testx, testy)     

    print("\n--Resultado KNN--")
    print("Tasa_clas: ", tasa_clas4)
    print("Funcion objetivo: ", funcion_objetivo4)
    print("Tiempo de ejecucion: ", tiempo4, "seg")
    
    tasa_red5, funcion_objetivo5, tasa_clas5, tiempo5 = greedy(trainx, trainy, testx, testy)

    print("\n--Resultado Greedy--")
    print("Tasa_red: ", tasa_red5)
    print("Tasa_clas: ", tasa_clas5)
    print("Funcion objetivo: ", funcion_objetivo5)
    print("Tiempo de ejecucion: ", tiempo5, "seg") 
    
    tasa_red6, funcion_objetivo6, tasa_clas6, tiempo6 = busqueda_local(trainx, trainy, testx, testy)

    print("\n--Resultado BL--")
    print("Tasa_red: ", tasa_red6)
    print("Tasa_clas: ", tasa_clas6)
    print("Funcion objetivo: ", funcion_objetivo6)
    print("Tiempo de ejecucion: ", tiempo6, "seg")
    
    
    ## En el AGG y AGE se modifica la ultima entrada para alternar entre BLX y CA
    tasa_redAGG2, funcion_objetivoAGG2, tasa_clasAGG2, tiempoAGG2 = AGG(trainx, trainy, testx, testy,'BLX')
    
    print("\n--Resultado AGG--")
    print("Tasa_red: ", tasa_redAGG2)
    print("Tasa_clas: ", tasa_clasAGG2)
    print("Funcion objetivo: ", funcion_objetivoAGG2)
    print("Tiempo de ejecucion: ", tiempoAGG2, "seg")
    
    
    tasa_redAGE2, funcion_objetivoAGE2, tasa_clasAGE2, tiempoAGE2 = AGE(trainx, trainy, testx, testy,'BLX')
    
    print("\n--Resultado AGE--")
    print("Tasa_red: ", tasa_redAGE2)
    print("Tasa_clas: ", tasa_clasAGE2)
    print("Funcion objetivo: ", funcion_objetivoAGE2)
    print("Tiempo de ejecucion: ", tiempoAGE2, "seg")
    
    
    tasa_redAM_1_0_2, funcion_objetivoAM_1_0_2, tasa_clasAM_1_0_2, tiempoAM_1_0_2 = AM(trainx, trainy, testx, testy,0)
    
    print("\n--Resultado AM_1.0--")
    print("Tasa_red: ", tasa_redAM_1_0_2)
    print("Tasa_clas: ", tasa_clasAM_1_0_2)
    print("Funcion objetivo: ", funcion_objetivoAM_1_0_2)
    print("Tiempo de ejecucion: ", tiempoAM_1_0_2, "seg")
    
    
    tasa_redAM_0_1_2, funcion_objetivoAM_0_1_2, tasa_clasAM_0_1_2, tiempoAM_0_1_2 = AM(trainx, trainy, testx, testy,1)
    
    print("\n--Resultado AM_0.1--")
    print("Tasa_red: ", tasa_redAM_0_1_2)
    print("Tasa_clas: ", tasa_clasAM_0_1_2)
    print("Funcion objetivo: ", funcion_objetivoAM_0_1_2)
    print("Tiempo de ejecucion: ", tiempoAM_0_1_2, "seg")
    
    
    tasa_redAM_0_1_mejores_2, funcion_objetivoAM_0_1_mejores_2, tasa_clasAM_0_1_mejores_2, tiempoAM_0_1_mejores_2 = AM(trainx, trainy, testx, testy,2)
    
    print("\n--Resultado AM_0.1_mejores--")
    print("Tasa_red: ", tasa_redAM_0_1_mejores_2)
    print("Tasa_clas: ", tasa_clasAM_0_1_mejores_2)
    print("Funcion objetivo: ", funcion_objetivoAM_0_1_mejores_2)
    print("Tiempo de ejecucion: ", tiempoAM_0_1_mejores_2, "seg")
    
    it+=1


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

print("\n\n## TEXTURE ##")
   
it = 1   
agr_103 = agr_013 = agr_01m3 = 0   
for i3, t3 in par.split(x3,y3):  
    print("\n\n**Particion ", it, "**")
    
    ## Divido la particion en train y test
    trainx, trainy, testx, testy = prepara_particiones(i3,t3,x3,y3) 
    
    tiempo7, tasa_clas7, funcion_objetivo7 = Knn(trainx, trainy, testx, testy)     

    print("\n--Resultado KNN--")
    print("Tasa_clas: ", tasa_clas7)
    print("Funcion objetivo: ", funcion_objetivo7)
    print("Tiempo de ejecucion: ", tiempo7, "seg")

    tasa_red8, funcion_objetivo8, tasa_clas8, tiempo8 = greedy(trainx, trainy, testx, testy)

    print("\n\n--Resultado Greedy--")
    print("Tasa_red: ", tasa_red8)
    print("Tasa_clas: ", tasa_clas8)
    print("Funcion objetivo: ", funcion_objetivo8)
    print("Tiempo de ejecucion: ", tiempo8, "seg")
    

    tasa_red9, funcion_objetivo9, tasa_clas9, tiempo9 = busqueda_local(trainx, trainy, testx, testy)

    print("\n--Resultado BL--")
    print("Tasa_red: ", tasa_red9)
    print("Tasa_clas: ", tasa_clas9)
    print("Funcion objetivo: ", funcion_objetivo9)
    print("Tiempo de ejecucion: ", tiempo9, "seg")
    
    
    ## En el AGG y AGE se modifica la ultima entrada para alternar entre BLX y CA
    tasa_redAGG3, funcion_objetivoAGG3, tasa_clasAGG3, tiempoAGG3 = AGG(trainx, trainy, testx, testy,'BLX')
    
    print("\n--Resultado AGG--")
    print("Tasa_red: ", tasa_redAGG3)
    print("Tasa_clas: ", tasa_clasAGG3)
    print("Funcion objetivo: ", funcion_objetivoAGG3)
    print("Tiempo de ejecucion: ", tiempoAGG3, "seg")
    
    
    tasa_redAGE3, funcion_objetivoAGE3, tasa_clasAGE3, tiempoAGE3 = AGE(trainx, trainy, testx, testy,'BLX')
    
    print("\n--Resultado AGE--")
    print("Tasa_red: ", tasa_redAGE3)
    print("Tasa_clas: ", tasa_clasAGE3)
    print("Funcion objetivo: ", funcion_objetivoAGE3)
    print("Tiempo de ejecucion: ", tiempoAGE3, "seg")
    
    
    
    tasa_redAM_1_0_3, funcion_objetivoAM_1_0_3, tasa_clasAM_1_0_3, tiempoAM_1_0_3 = AM(trainx, trainy, testx, testy,0)
    
    print("\n--Resultado AM_1.0--")
    print("Tasa_red: ", tasa_redAM_1_0_3)
    print("Tasa_clas: ", tasa_clasAM_1_0_3)
    print("Funcion objetivo: ", funcion_objetivoAM_1_0_3)
    print("Tiempo de ejecucion: ", tiempoAM_1_0_3, "seg")
    
    
    tasa_redAM_0_1_3, funcion_objetivoAM_0_1_3, tasa_clasAM_0_1_3, tiempoAM_0_1_3 = AM(trainx, trainy, testx, testy,1)
    
    print("\n--Resultado AM_0.1--")
    print("Tasa_red: ", tasa_redAM_0_1_3)
    print("Tasa_clas: ", tasa_clasAM_0_1_3)
    print("Funcion objetivo: ", funcion_objetivoAM_0_1_3)
    print("Tiempo de ejecucion: ", tiempoAM_0_1_3, "seg")
    
    
    tasa_redAM_0_1_mejores_3, funcion_objetivoAM_0_1_mejores_3, tasa_clasAM_0_1_mejores_3, tiempoAM_0_1_mejores_3 = AM(trainx, trainy, testx, testy,2)
    
    print("\n--Resultado AM_0.1_mejores--")
    print("Tasa_red: ", tasa_redAM_0_1_mejores_3)
    print("Tasa_clas: ", tasa_clasAM_0_1_mejores_3)
    print("Funcion objetivo: ", funcion_objetivoAM_0_1_mejores_3)
    print("Tiempo de ejecucion: ", tiempoAM_0_1_mejores_3, "seg")
    
    it +=1
