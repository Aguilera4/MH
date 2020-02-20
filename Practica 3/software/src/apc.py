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
import matplotlib.pyplot as plt
import time


## Fijo la semilla 
np.random.seed(2)


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
    
    funcion_objetivo, tasa_clas, tasa_red = evaluacion(w, trainx, trainy)
    
    ## Iteramos hasta llegar al maximo de vecinos o al maximo de iteraciones
    while vecinos < 20 * len(trainx[0]) and  it < 15000:
        
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
                       
            val, tasa_clas, tasa_red = evaluacion(w, trainx, trainy)
            
            # Si valor de la funcion acutal es mayor que la global actualizamos y establecemos vecinos a 0, si no
            # recuperamos el vector de pesos anterior y seguimos mutando.
            if val > funcion_objetivo:
                funcion_objetivo = val
                vecinos = 0
                break
            else:
                w[l] = w_aux
                vecinos += 1
                
    
    funcion_objetivo , tasa_clas, tasa_red = calcula_evaluacion_final(w, trainx, trainy, testx, testy)

    ## Control de tiempo
    tiempo = time.time() - inicio
             
    return tasa_red, funcion_objetivo, tasa_clas, tiempo
                  
################################# PRÁCTICA 2 ##################################


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


################################# PRÁCTICA 3 ##################################

# Evaluación sobre datos train
def evaluacion(w, trainx, trainy):
    # Clasificador 1NN
    w_trunc = w.copy()
    cont_reduccion = np.count_nonzero(w_trunc<0.2)
    w_trunc[w_trunc<0.2] = 0.0
    trainx_a = (trainx * w_trunc)
    
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)   
    ind_near = clasificador.kneighbors(trainx_a, n_neighbors=2)[1][:,1] 
    funcion_objetivo, tasa_clas, tasa_red = calcula_funcion_objetivo(ind_near, trainy, cont_reduccion, w)
    
    return funcion_objetivo, tasa_clas, tasa_red

# Evaluación final sobre datos test
def calcula_evaluacion_final(w, trainx, trainy, testx, testy):
    # Caculamos los valores de tasa_red, tasa_clas y función objetivo final
    cont_reduccion = np.count_nonzero(w<0.2) 
    w[w<0.2] = 0.0             
    trainx_a = (trainx * w)  
    testx_a = (testx * w)
    
     # Clasificador 1NN
    clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clasificador.fit(trainx_a,trainy)
    prediccion = clasificador.predict(testx_a)
    
    # Cálculo el porcentaje de aciertos
    tasa_clas = metrics.accuracy_score(testy,prediccion) * 100
    tasa_red = cont_reduccion / len(w) * 100
                    
    ## Funcion objetivo
    funcion_objetivo = (tasa_clas + tasa_red) * 0.5
    
    return funcion_objetivo, tasa_clas, tasa_red


# Obtiene tantos indices de padres como indique num_elegir, los indices no son
# se seleccionan de forma repetida
def obtener_indices_padres(num_elegir, num_pobla):
    indices_padres = []
    cont = 0
    while cont < num_elegir:
        indice = np.random.randint(0,num_pobla)
        if (indice in indices_padres) == False:
            indices_padres.append(indice)
            cont += 1
            
    return indices_padres


    
######################      Enfriamiento Simulado     #########################

def efriamiento_simulado(trainx, trainy, testx, testy):
    ## Control de tiempo
    inicio = time.time()
    
    val_convergenciaES = []
    it_convergenciaES = []
    
    # Solucion inicial aleatoria 
    w_actual = np.random.uniform(0,1,len(trainx[0]))
    
    # Inicializacion de los parametros
    max_vecinos = 10 * len(w_actual)
    max_exitos = 0.1 * max_vecinos
    M = round(15000/max_vecinos)
    vecinos = 0
    num_exitos = 1
    it = 0
    k = 1
    
    val, tasa_clas, tasa_red = evaluacion(w_actual, trainx, trainy)
    
    mejor_w = w_actual.copy()
    mejor_val = val
    
    # Temperatura inicial y final
    T_inicial = (0.3 * (val))/(-np.log(0.3))

    T_fin = 0.001    
    while T_fin > T_inicial:
        T_fin = T_fin * 0.001
    
    # Calculamos beta       
    beta = (T_inicial - T_fin)/(M*T_fin*T_inicial)
    
    # Inicializamos temperatura actual = T_inicial
    T_actual = T_inicial
    i = 0
    # Bucle externo
    while it < M and num_exitos > 0 and T_actual > T_fin:
        num_exitos = 0
        vecinos = 0
        
        # Bucle interno
        while num_exitos < max_exitos and vecinos < max_vecinos:
            w_nueva = w_actual.copy()
            
            ## Cambio por mutación normal
            z = np.random.normal(0.0,0.3)
            indice = np.random.randint(0,len(w_actual),1)
            w_nueva[indice] += z
    
            if w_nueva[indice] < 0.0:
                w_nueva[indice] = 0.0
            if w_nueva[indice] > 1.0:
                w_nueva[indice] = 1.0
                
            # Evaluación mutación
            val_nueva, tasa_clas, tasa_red = evaluacion(w_nueva, trainx, trainy)
            # Calculo de deltaF
            diferencia = val_nueva - val
            
            
           
            
            if diferencia > 0 or np.random.uniform(0.0,1.0) <= np.exp((diferencia)/(k*T_actual)):
                val = val_nueva
                w_actual = w_nueva.copy()
                num_exitos += 1
                if val > mejor_val: 
                    mejor_val = val
                    mejor_w = w_actual

            vecinos += 1
        
        # Vectores utilizados para el estudio de la convergencia (no influye en el algoritmo)
        val_convergenciaES.append(mejor_val)
        it_convergenciaES.append(i)
        i += 1
        # calculamos temperatura actual    
        T_actual = T_actual / (1 + (beta * T_actual))
        it += 1
    
    # Evaluacion final sobre test
    funcion_objetivo , tasa_clas, tasa_red = calcula_evaluacion_final(mejor_w, trainx, trainy, testx, testy)

    ## Control de tiempo
    tiempo = time.time() - inicio
             
    return tasa_red, funcion_objetivo, tasa_clas, tiempo, val_convergenciaES, it_convergenciaES
        
    
#############################      ILS       ##################################
def ILS(trainx, trainy, testx, testy):
    ## Control de tiempo
    inicio = time.time()
    val_convergenciaILS = []
    it_convergenciaILS = []
    
    # Generación de la solución inicial
    w_actual = np.random.uniform(0,1,len(trainx[0]))
    it = 0
    num_mutaciones = round(0.1 * len(trainx[0]))
    
    val, tasa_clas, tasa_red = evaluacion(w_actual, trainx, trainy)
    
    # Búsqueda local sobre la solución inicial
    val_actual, w_actual = busqueda_local_ILS(trainx,trainy,w_actual)
    it += 1
    
    mejor_w = w_actual.copy()
    mejor_val = val_actual
    
    val_convergenciaILS.append(mejor_val)
    it_convergenciaILS.append(it)
    
    # Bucle externo
    while it < 15:
        ## Cambio por mutacion normal
        for t in range(num_mutaciones):
            gen = np.random.randint(0,len(w_actual))
            z = np.random.normal(0.0,0.4)
            w_actual[gen] += z

            if w_actual[gen] < 0.0:
                w_actual[gen] = 0.0
            if w_actual[gen] > 1.0:
                w_actual[gen] = 1.0
        
        # Búsqueda local sobre la solución actual
        val_actual, w_actual = busqueda_local_ILS(trainx,trainy,w_actual)
        #Evalucación
        val_actual, tasa_clas, tasa_red = evaluacion(w_actual, trainx, trainy)
        
        # actualizamos la mejor solución
        if val_actual > mejor_val:
            mejor_w = w_actual
            mejor_val = val_actual
        
        # Vectores utilizados para el estudio de la convergencia (no influye en el algoritmo)
        val_convergenciaILS.append(mejor_val)
        it_convergenciaILS.append(it)
        
        it += 1    
        w_actual = mejor_w.copy()
    # Evaluacion final sobre test
    funcion_objetivo , tasa_clas, tasa_red = calcula_evaluacion_final(mejor_w, trainx, trainy, testx, testy)

    ## Control de tiempo
    tiempo = time.time() - inicio
                    
    return tasa_red, funcion_objetivo, tasa_clas, tiempo, val_convergenciaILS, it_convergenciaILS
    


#########################   Evolución Diferencial Rand  #######################
def evolucion_diferencial_rand(trainx, trainy, testx, testy):
    ## Control de tiempo
    inicio = time.time()
    
    val_convergenciaDErand = []
    it_convergenciaDErand = []
    
    it = 0
    # Genero la poblacion inicial y la evaluo
    poblacion = genera_poblacion(trainx,50)
    eval_poblacion, it = evalua_poblacion(trainx,trainy,poblacion, it)
    
    # Bucle externo
    while it < 15000:
        
        for i in range(len(poblacion)):
            # obtenemos los indices de tres padres aleatoriamente            
            indices_padres = obtener_indices_padres(3, len(poblacion))

            gen = np.random.randint(0, len(poblacion[0]))
            # Recombinación
            hijo = np.zeros(len(poblacion[0]))
            for j in range(len(poblacion[i])):
                prob = np.random.uniform(0,1)
                if prob < 0.5 or j == gen:
                    # generamos Offspring_ij
                    valor = poblacion[indices_padres[0]][j] + 0.5 * (poblacion[indices_padres[1]][j] - poblacion[indices_padres[2]][j])
                    # Normalizamos
                    if valor < 0.0:
                        valor = 0.0
                    elif valor > 1.0:
                        valor = 1.0
                     
                    hijo[j] = valor
                else:
                    hijo[j] = poblacion[i][j]
            
            # Evaluacion del hijo
            eval_hijo, _, _ = evaluacion(hijo, trainx, trainy)            
            it += 1
            
            # Reemplazamiento
            if eval_hijo > eval_poblacion[i]:
               poblacion[i] = hijo
               eval_poblacion[i] = eval_hijo 
               
        # Vectores utilizados para el estudio de la convergencia (no influye en el algoritmo)
        id_max = eval_poblacion.index(max(eval_poblacion)) 
        val_convergenciaDErand.append(eval_poblacion[id_max])
        it_convergenciaDErand.append(it)
    
    # Evaluacion final sobre test
    funcion_objetivo , tasa_clas, tasa_red = calcula_evaluacion_final(poblacion[id_max], trainx, trainy, testx, testy)
    
    ## Control de tiempo
    tiempo = time.time() - inicio
    
    return tasa_red, funcion_objetivo, tasa_clas, tiempo, val_convergenciaDErand, it_convergenciaDErand




###################  Evolucion diferencial current-to-best  ###################
    
def evolucion_diferencial_current_to_best(trainx, trainy, testx, testy):
    ## Control de tiempo
    inicio = time.time()
    it = 0
    
    val_convergenciaDEcurrentToBest = []
    it_convergenciaDEcurrentToBest = []
    
    # Genero la poblacion inicial y la evaluo
    poblacion = genera_poblacion(trainx,50)
    eval_poblacion, it = evalua_poblacion(trainx,trainy,poblacion, it)
    
    # Bucle externo
    while it < 15000:
        
        for i in range(len(poblacion)):
            
            id_max = eval_poblacion.index(max(eval_poblacion))
            mejor_padre = poblacion[id_max]
            
            # obtenemos los indices de dos padres aleatoriamente    
            indices_padres = obtener_indices_padres(2, len(poblacion))
                        
            hijo = np.zeros(len(poblacion[0]))
            # Recombinación
            for j in range(len(poblacion[0])):
                prob = np.random.uniform(0,1)
                if prob < 0.5:
                    # generamos Offspring_ij
                    valor = poblacion[i][j] + 0.5 * (mejor_padre[j] - poblacion[i][j]) + 0.5 * (poblacion[indices_padres[0]][j] - poblacion[indices_padres[1]][j])
                    # Normalizamos
                    if valor < 0.0:
                        valor = 0.0
                    elif valor > 1.0:
                        valor = 1.0
                     
                    hijo[j] = valor
                else:
                    hijo[j] = poblacion[i][j]
             
            # Evaluacion del hijo
            eval_hijo,_,_ = evaluacion(hijo, trainx, trainy)            
            it += 1
            
            # Reemplazamiento
            if eval_hijo > eval_poblacion[i]:
               poblacion[i] = hijo
               eval_poblacion[i] = eval_hijo 
        
        # Vectores utilizados para el estudio de la convergencia (no influye en el algoritmo)
        id_max = eval_poblacion.index(max(eval_poblacion))
        val_convergenciaDEcurrentToBest.append(eval_poblacion[id_max])
        it_convergenciaDEcurrentToBest.append(it)         
                

    # Selecciono el individuo con mayor valor de funcion objetivo
    id_max = eval_poblacion.index(max(eval_poblacion))
    
    # Evaluacion final sobre test
    funcion_objetivo , tasa_clas, tasa_red = calcula_evaluacion_final(poblacion[id_max], trainx, trainy, testx, testy)
    
    ## Control de tiempo
    tiempo = time.time() - inicio
    
    return tasa_red, funcion_objetivo, tasa_clas, tiempo, val_convergenciaDEcurrentToBest, it_convergenciaDEcurrentToBest




#############################      BL-ILS      ################################

def busqueda_local_ILS(trainx, trainy,w):       
    
    vecinos = 0
    it = 0
    
    funcion_objetivo, tasa_clas, tasa_red = evaluacion(w, trainx, trainy)
    
    ## Iteramos hasta llegar al maximo de vecinos o al maximo de iteraciones
    while vecinos < 20 * len(trainx[0]) and  it < 1000:
          
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
                       
            val, tasa_clas, tasa_red = evaluacion(w, trainx, trainy)
            
            # Si valor de la funcion acutal es mayor que la global actualizamos y establecemos vecinos a 0, si no
            # recuperamos el vector de pesos anterior y seguimos mutando.
            if val > funcion_objetivo:
                funcion_objetivo = val
                vecinos = 0
                break
            else:
                w[l] = w_aux
                vecinos += 1
             
    return funcion_objetivo, w


# --------------------- MAIN ---------------------------------------------



def plot_convergencia(x,y,title,xlabel,ylabel):
    color = ['r','b','k','g','m']
    for i in range(5):
        plt.plot(y[i], x[i], c=color[i], label = 'Particion'+ str(i+1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


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
valTotal_convergenciaES = []
itTotal_convergenciaES = []

valTotal_convergenciaILS = []
itTotal_convergenciaILS = []

valTotal_convergenciaDERand = []
itTotal_convergenciaDERand = []

valTotal_convergenciaDEcurrent = []
itTotal_convergenciaDEcurrent= []


agr_10 = agr_01 = agr_01m = 0
for i, t in par.split(x,y):
    print("\n\n**Particion ", it, "**")
    
    ## Divido la particion en train y test
    trainx, trainy, testx, testy = prepara_particiones(i,t,x,y)  
    
    
    tasa_red_ES, funcion_objetivo_ES, tasa_clas_ES, tiempo_ES, val_convergenciaES, it_convergenciaES = efriamiento_simulado(trainx, trainy, testx, testy)
    
    print("\n--Resultado Enfriamiento Simulado--")
    print("Tasa_red: ", tasa_red_ES)
    print("Tasa_clas: ", tasa_clas_ES)
    print("Funcion objetivo: ", funcion_objetivo_ES)
    print("Tiempo de ejecucion: ", tiempo_ES, "seg")
    
    valTotal_convergenciaES.append(val_convergenciaES)
    itTotal_convergenciaES.append(it_convergenciaES)
    
    
    tasa_red_ILS, funcion_objetivo_ILS, tasa_clas_ILS, tiempo_ILS, val_convergenciaILS, it_convergenciaILS = ILS(trainx, trainy, testx, testy)
    
    print("\n--Resultado ILS--")
    print("Tasa_red: ", tasa_red_ILS)
    print("Tasa_clas: ", tasa_clas_ILS)
    print("Funcion objetivo: ", funcion_objetivo_ILS)
    print("Tiempo de ejecucion: ", tiempo_ILS, "seg")
    
    valTotal_convergenciaILS.append(val_convergenciaILS)
    itTotal_convergenciaILS.append(it_convergenciaILS)
    
    
    tasa_red_ED, funcion_objetivo_ED, tasa_clas_ED, tiempo_ED, val_convergenciaDErand, it_convergenciaDErand = evolucion_diferencial_rand(trainx, trainy, testx, testy)
    
    print("\n--Resultado ED-Rand--")
    print("Tasa_red: ", tasa_red_ED)
    print("Tasa_clas: ", tasa_clas_ED)
    print("Funcion objetivo: ", funcion_objetivo_ED)
    print("Tiempo de ejecucion: ", tiempo_ED, "seg")
    
    valTotal_convergenciaDERand.append(val_convergenciaDErand)
    itTotal_convergenciaDERand.append(it_convergenciaDErand)
    
    
    tasa_red_ED_mejor, funcion_objetivo_ED_mejor, tasa_clas_ED_mejor, tiempo_ED_mejor, val_convergenciaDEcurrent, it_convergenciaDEcurrent = evolucion_diferencial_current_to_best(trainx, trainy, testx, testy)
    
    print("\n--Resultado ED-Current-to-best--")
    print("Tasa_red: ", tasa_red_ED_mejor)
    print("Tasa_clas: ", tasa_clas_ED_mejor)
    print("Funcion objetivo: ", funcion_objetivo_ED_mejor)
    print("Tiempo de ejecucion: ", tiempo_ED_mejor, "seg")
    
    valTotal_convergenciaDEcurrent.append(val_convergenciaDEcurrent)
    itTotal_convergenciaDEcurrent.append(it_convergenciaDEcurrent)
    
    
    it +=1



plot_convergencia(valTotal_convergenciaES, itTotal_convergenciaES, 'Convergencia ES (Colposcopy)', 'Enfriamientos', 'Valor agr')
plot_convergencia(valTotal_convergenciaILS, itTotal_convergenciaILS, 'Convergencia ILS (Colposcopy)', 'Iteraciones', 'Valor agr')
plot_convergencia(valTotal_convergenciaDERand, itTotal_convergenciaDERand, 'Convergencia DE/Rand/1 (Colposcopy)', 'Iteraciones', 'Valor agr')
plot_convergencia(valTotal_convergenciaDEcurrent, itTotal_convergenciaDEcurrent, 'Convergencia DE/current-to-best/1 (Colposcopy)', 'Iteraciones', 'Valor agr')

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

valTotal_convergenciaES = []
itTotal_convergenciaES = []

valTotal_convergenciaILS = []
itTotal_convergenciaILS = []

valTotal_convergenciaDERand = []
itTotal_convergenciaDERand = []

valTotal_convergenciaDEcurrent = []
itTotal_convergenciaDEcurrent= []


agr_102 = agr_012 = agr_01m2 = 0
for i2, t2 in par.split(x2,y2):
    print("\n\n**Particion ", it, "**")
    
    ## Divido la particion en train y test
    trainx, trainy, testx, testy = prepara_particiones(i2,t2,x2,y2) 

    
    tasa_red_ES_2, funcion_objetivo_ES_2, tasa_clas_ES_2, tiempo_ES_2, val_convergenciaES, it_convergenciaES = efriamiento_simulado(trainx, trainy, testx, testy)
    
    print("\n--Resultado Enfriamiento Simulado--")
    print("Tasa_red: ", tasa_red_ES_2)
    print("Tasa_clas: ", tasa_clas_ES_2)
    print("Funcion objetivo: ", funcion_objetivo_ES_2)
    print("Tiempo de ejecucion: ", tiempo_ES_2, "seg")
    
    valTotal_convergenciaES.append(val_convergenciaES)
    itTotal_convergenciaES.append(it_convergenciaES)
    
    
    
    tasa_red_ILS_2, funcion_objetivo_ILS_2, tasa_clas_ILS_2, tiempo_ILS_2, val_convergenciaILS, it_convergenciaILS = ILS(trainx, trainy, testx, testy)
    
    print("\n--Resultado ILS--")
    print("Tasa_red: ", tasa_red_ILS_2)
    print("Tasa_clas: ", tasa_clas_ILS_2)
    print("Funcion objetivo: ", funcion_objetivo_ILS_2)
    print("Tiempo de ejecucion: ", tiempo_ILS_2, "seg")
    
    valTotal_convergenciaILS.append(val_convergenciaILS)
    itTotal_convergenciaILS.append(it_convergenciaILS)
    
    
    
    tasa_red_ED_2, funcion_objetivo_ED_2, tasa_clas_ED_2, tiempo_ED_2, val_convergenciaDErand, it_convergenciaDErand = evolucion_diferencial_rand(trainx, trainy, testx, testy)
    
    print("\n--Resultado ED-Rand--")
    print("Tasa_red: ", tasa_red_ED_2)
    print("Tasa_clas: ", tasa_clas_ED_2)
    print("Funcion objetivo: ", funcion_objetivo_ED_2)
    print("Tiempo de ejecucion: ", tiempo_ED_2, "seg")
    
    valTotal_convergenciaDERand.append(val_convergenciaDErand)
    itTotal_convergenciaDERand.append(it_convergenciaDErand)
    
    
    tasa_red_ED_mejor_2, funcion_objetivo_ED_mejor_2, tasa_clas_ED_mejor_2, tiempo_ED_mejor_2, val_convergenciaDEcurrent, it_convergenciaDEcurrent = evolucion_diferencial_current_to_best(trainx, trainy, testx, testy)
    
    print("\n--Resultado ED-Current-to-best--")
    print("Tasa_red: ", tasa_red_ED_mejor_2)
    print("Tasa_clas: ", tasa_clas_ED_mejor_2)
    print("Funcion objetivo: ", funcion_objetivo_ED_mejor_2)
    print("Tiempo de ejecucion: ", tiempo_ED_mejor_2, "seg")
    
    valTotal_convergenciaDEcurrent.append(val_convergenciaDEcurrent)
    itTotal_convergenciaDEcurrent.append(it_convergenciaDEcurrent)
    
    
    it+=1

plot_convergencia(valTotal_convergenciaES, itTotal_convergenciaES, 'Convergencia ES (Ionosphere)', 'Enfriamientos', 'Valor agr')
plot_convergencia(valTotal_convergenciaILS, itTotal_convergenciaILS, 'Convergencia ILS (Ionosphere)', 'Iteraciones', 'Valor agr')
plot_convergencia(valTotal_convergenciaDERand, itTotal_convergenciaDERand, 'Convergencia DE/Rand/1 (Ionosphere)', 'Iteraciones', 'Valor agr')
plot_convergencia(valTotal_convergenciaDEcurrent, itTotal_convergenciaDEcurrent, 'Convergencia DE/current-to-best/1 (Ionosphere)', 'Iteraciones', 'Valor agr')


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

valTotal_convergenciaES = []
itTotal_convergenciaES = []

valTotal_convergenciaILS = []
itTotal_convergenciaILS = []

valTotal_convergenciaDERand = []
itTotal_convergenciaDERand = []

valTotal_convergenciaDEcurrent = []
itTotal_convergenciaDEcurrent= []

it = 1   
agr_103 = agr_013 = agr_01m3 = 0   
for i3, t3 in par.split(x3,y3):  
    print("\n\n**Particion ", it, "**")
    
    ## Divido la particion en train y test
    trainx, trainy, testx, testy = prepara_particiones(i3,t3,x3,y3) 
    
    
    tasa_red_ES_3, funcion_objetivo_ES_3, tasa_clas_ES_3, tiempo_ES_3, val_convergenciaES, it_convergenciaES = efriamiento_simulado(trainx, trainy, testx, testy)
    
    print("\n--Resultado Enfriamiento Simulado--")
    print("Tasa_red: ", tasa_red_ES_3)
    print("Tasa_clas: ", tasa_clas_ES_3)
    print("Funcion objetivo: ", funcion_objetivo_ES_3)
    print("Tiempo de ejecucion: ", tiempo_ES_3, "seg")
    
    valTotal_convergenciaES.append(val_convergenciaES)
    itTotal_convergenciaES.append(it_convergenciaES)
    
    
    tasa_red_ILS_3, funcion_objetivo_ILS_3, tasa_clas_ILS_3, tiempo_ILS_3, val_convergenciaILS, it_convergenciaILS = ILS(trainx, trainy, testx, testy)
    
    print("\n--Resultado ILS--")
    print("Tasa_red: ", tasa_red_ILS_3)
    print("Tasa_clas: ", tasa_clas_ILS_3)
    print("Funcion objetivo: ", funcion_objetivo_ILS_3)
    print("Tiempo de ejecucion: ", tiempo_ILS_3, "seg")
    
    valTotal_convergenciaILS.append(val_convergenciaILS)
    itTotal_convergenciaILS.append(it_convergenciaILS)
    
    
    
    tasa_red_ED_3, funcion_objetivo_ED_3, tasa_clas_ED_3, tiempo_ED_3, val_convergenciaDErand, it_convergenciaDErand = evolucion_diferencial_rand(trainx, trainy, testx, testy)
    
    print("\n--Resultado ED-Rand--")
    print("Tasa_red: ", tasa_red_ED_3)
    print("Tasa_clas: ", tasa_clas_ED_3)
    print("Funcion objetivo: ", funcion_objetivo_ED_3)
    print("Tiempo de ejecucion: ", tiempo_ED_3, "seg")
    
    valTotal_convergenciaDERand.append(val_convergenciaDErand)
    itTotal_convergenciaDERand.append(it_convergenciaDErand)
    
    
    tasa_red_ED_mejor_3, funcion_objetivo_ED_mejor_3, tasa_clas_ED_mejor_3, tiempo_ED_mejor_3, val_convergenciaDEcurrent, it_convergenciaDEcurrent = evolucion_diferencial_current_to_best(trainx, trainy, testx, testy)
    
    print("\n--Resultado ED-Current-to-best--")
    print("Tasa_red: ", tasa_red_ED_mejor_3)
    print("Tasa_clas: ", tasa_clas_ED_mejor_3)
    print("Funcion objetivo: ", funcion_objetivo_ED_mejor_3)
    print("Tiempo de ejecucion: ", tiempo_ED_mejor_3, "seg")
    
    valTotal_convergenciaDEcurrent.append(val_convergenciaDEcurrent)
    itTotal_convergenciaDEcurrent.append(it_convergenciaDEcurrent)
    
    it +=1

plot_convergencia(valTotal_convergenciaES, itTotal_convergenciaES, 'Convergencia ES (Texture)', 'Enfriamientos', 'Valor agr')
plot_convergencia(valTotal_convergenciaILS, itTotal_convergenciaILS, 'Convergencia ILS (Texture)', 'Iteraciones', 'Valor agr')
plot_convergencia(valTotal_convergenciaDERand, itTotal_convergenciaDERand, 'Convergencia DE/Rand/1 (Texture)', 'Iteraciones', 'Valor agr')
plot_convergencia(valTotal_convergenciaDEcurrent, itTotal_convergenciaDEcurrent, 'Convergencia DE/current-to-best/1 (Texture)', 'Iteraciones', 'Valor agr')