#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Librerias:
# Importamos las librerias que se utilizarán para desarrollar el AG.
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import heapq
import time
import math
import cv2
import sys
import pickle
import os 

#-------------------------------1.Estructura de los Genes--------------------------------#
class Centroid:
    #Definimos la clase que representará el centro del círculo.
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
class RGB_Color:
    #Definimos la clase que representará el color del círculo.
    def __init__(self,r,g,b):
        self.r = r
        self.g = g
        self.b = b
        
class Gene:
    #Definimos la estructura del Gen como las caraterísticas de un círculo (posición, diametro y color).
    def __init__(self,size):
        self.size = (200,200)
        self.diameter = random.randint(10,20)
        self.position = Centroid(random.randint(0,size[0]),random.randint(0,size[1]))
        self.color = RGB_Color(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        self.params = ["diameter","position","color"]

#-------------------------------2.Estructura de los Cromosomas--------------------------------#

class Chromosome(object):   
   
    def __init__(self, chromosome):
            self.chromosome = chromosome[:]
            self.fitness = -1
    
    def Paint_Image(self):
        #Definimos la función para generar la imágen.
        x_max = self.chromosome[0].size[0]
        y_max = self.chromosome[0].size[1]
        img = np.zeros((x_max, y_max, 3), np.uint8)
        for i in range(len(self.chromosome)):
            color = (self.chromosome[i].color.r,self.chromosome[i].color.g,self.chromosome[i].color.b)
            center = (self.chromosome[i].position.x,self.chromosome[i].position.y)
            radius = int(self.chromosome[i].diameter/2)
            if i == 0:
                image = cv2.circle(img, center, radius, color, thickness=-1, lineType=8, shift=0)
            else:
                image = cv2.circle(image, center, radius, color, thickness=-1, lineType=8, shift=0) 
        
        return image    
             
    def crossover_onepoint(self, other):
        #Definimos la función de cruzamiento "OnePoint".
        #Obtenemos el punto de corte de forma aleatoria, tomando en cuenta el tamaño del cromosoma más pequeño para realizar
        #correctamente el cruzamiento.
        cut = random.randrange(len(self.chromosome))
        if cut > len(other.chromosome):
            cut = len(other.chromosome)
            
        child1 = Chromosome(self.chromosome[:cut] + other.chromosome[cut:])
        child2 = Chromosome(other.chromosome[:cut] + self.chromosome[cut:])
        
        return [child1, child2]

    def crossover_uniform(self, other):        
        #Definimos la función de cruzamiento "Uniform".
        chromosome1 = deepcopy(self.chromosome)
        chromosome2 = deepcopy(other.chromosome)
        #Considerar que el cruzamiento se realizará entre los genes que se encuentren dentro del rango del tamaño del cromosoma
        #más pequeño, para asi realizar correctamente el cruzamiento.        
        if (len(self.chromosome) > len(other.chromosome)):
            top_size = deepcopy(other.chromosome)
        else:
            top_size = deepcopy(self.chromosome)
            
        for i in range(len(top_size)):
            if random.uniform(0, 1) < 0.5:
                chromosome1[i] = self.chromosome[i]
                chromosome2[i] = other.chromosome[i]
            else:
                chromosome1[i] = other.chromosome[i]
                chromosome2[i] = self.chromosome[i]
                
        child1 = Chromosome(chromosome1)
        child2 = Chromosome(chromosome2)
        
        return [child1, child2] 

    def crossover_arithmetic_unique(self, other):
        #Definimos la función de cruzamiento "Aritmético-Unique" que responde a la siguiente ecuación 
        #para un solo gen seleccionado de manera aleatoria:
        # chromosome1 = alpha*self + (1−alpha)*other
        # chromosome2 = alpha*other + (1−alpha)*self
        chromosome1 = deepcopy(self.chromosome)
        chromosome2 = deepcopy(other.chromosome)
        x_max = self.chromosome[0].size[0]
        y_max = self.chromosome[0].size[1]  
        alpha = random.uniform(0, 1)     
        #Obtenemos el gen de forma aleatoria, tomando en cuenta el tamaño del cromosoma más pequeño para realizar
        #correctamente el cruzamiento.       
        cut = random.randrange(len(self.chromosome))
        if cut >= len(other.chromosome):
            cut = (len(other.chromosome)-1)
        #Generamos el primer hijo:    
        chromosome1[cut].diameter = (alpha*self.chromosome[cut].diameter + (other.chromosome[cut].diameter)*(1-alpha)) #Modificamos el valor del diámetro en base a ambos padres.
        x = int(alpha*self.chromosome[cut].position.x + (other.chromosome[cut].position.x)*(1-alpha))
        y = int(alpha*self.chromosome[cut].position.y + (other.chromosome[cut].position.y)*(1-alpha))
        chromosome1[cut].position = Centroid(min(x,x_max),min(y,y_max)) #Modificamos el valor de la posición en base a ambos padres.
        r = int(alpha*self.chromosome[cut].color.r + (other.chromosome[cut].color.r)*(1-alpha))
        g = int(alpha*self.chromosome[cut].color.g + (other.chromosome[cut].color.g)*(1-alpha))
        b = int(alpha*self.chromosome[cut].color.b + (other.chromosome[cut].color.b)*(1-alpha))
        chromosome1[cut].color = RGB_Color(min(r,255),min(g,255),min(b,255)) #Modificamos el valor del color en base a ambos padres.
        #Generamos el segundo hijo:            
        chromosome2[cut].diameter = (alpha*other.chromosome[cut].diameter + (self.chromosome[cut].diameter)*(1-alpha)) #Modificamos el valor del diámetro en base a ambos padres.
        x = int(alpha*other.chromosome[cut].position.x + (self.chromosome[cut].position.x)*(1-alpha))
        y = int(alpha*other.chromosome[cut].position.y + (self.chromosome[cut].position.y)*(1-alpha))
        chromosome2[cut].position = Centroid(min(x,x_max),min(y,y_max)) #Modificamos el valor de la posición en base a ambos padres.
        r = int(alpha*other.chromosome[cut].color.r + (self.chromosome[cut].color.r)*(1-alpha))
        g = int(alpha*other.chromosome[cut].color.g + (self.chromosome[cut].color.g)*(1-alpha))
        b = int(alpha*other.chromosome[cut].color.b + (self.chromosome[cut].color.b)*(1-alpha))
        chromosome2[cut].color = RGB_Color(min(r,255),min(g,255),min(b,255)) #Modificamos el valor del color en base a ambos padres.
        
        child1 = Chromosome(chromosome1)
        child2 = Chromosome(chromosome2)
        
        return [child1, child2]    
    
    def crossover_arithmetic(self, other):
        #Definimos la función de cruzamiento "Aritmético-Complete" que responde a la siguiente ecuación 
        #para todos los genes de ambos padres:
        # chromosome1 = alpha*self + (1−alpha)*other
        # chromosome2 = alpha*other + (1−alpha)*self
        #Considerar que el cruzamiento se realizará entre los genes que se encuentren dentro del rango del tamaño del cromosoma
        #más pequeño, para asi realizar correctamente el cruzamiento. 
        if (len(self.chromosome) > len(other.chromosome)):
            top_size = deepcopy(other.chromosome)
        else:
            top_size = deepcopy(self.chromosome)
        
        chromosome1 = deepcopy(self.chromosome)
        chromosome2 = deepcopy(other.chromosome)
        x_max = self.chromosome[0].size[0]
        y_max = self.chromosome[0].size[1]
        alpha = random.uniform(0, 1)   
        
        for i in range(len(top_size)):          
            chromosome1[i].diameter = (alpha*self.chromosome[i].diameter + (other.chromosome[i].diameter)*(1-alpha)) #Modificamos el valor del diámetro en base a ambos padres.
            x = int(alpha*self.chromosome[i].position.x + (other.chromosome[i].position.x)*(1-alpha))
            y = int(alpha*self.chromosome[i].position.y + (other.chromosome[i].position.y)*(1-alpha))
            chromosome1[i].position = Centroid(min(x,x_max),min(y,y_max)) #Modificamos el valor de la posición en base a ambos padres.
            r = int(alpha*self.chromosome[i].color.r + (other.chromosome[i].color.r)*(1-alpha))
            g = int(alpha*self.chromosome[i].color.g + (other.chromosome[i].color.g)*(1-alpha))
            b = int(alpha*self.chromosome[i].color.b + (other.chromosome[i].color.b)*(1-alpha))
            chromosome1[i].color = RGB_Color(min(r,255),min(g,255),min(b,255)) #Modificamos el valor del color en base a ambos padres.
            
            chromosome2[i].diameter = (alpha*other.chromosome[i].diameter + (self.chromosome[i].diameter)*(1-alpha)) #Modificamos el valor del diámetro en base a ambos padres.
            x = int(alpha*other.chromosome[i].position.x + (self.chromosome[i].position.x)*(1-alpha))
            y = int(alpha*other.chromosome[i].position.y + (self.chromosome[i].position.y)*(1-alpha))
            chromosome2[i].position = Centroid(min(x,x_max),min(y,y_max)) #Modificamos el valor de la posición en base a ambos padres.
            r = int(alpha*other.chromosome[i].color.r + (self.chromosome[i].color.r)*(1-alpha))
            g = int(alpha*other.chromosome[i].color.g + (self.chromosome[i].color.g)*(1-alpha))
            b = int(alpha*other.chromosome[i].color.b + (self.chromosome[i].color.b)*(1-alpha))
            chromosome2[i].color = RGB_Color(min(r,255),min(g,255),min(b,255)) #Modificamos el valor del color en base a ambos padres.
        
        child1 = Chromosome(chromosome1)
        child2 = Chromosome(chromosome2)
        
        return [child1, child2]
    
    def mutate_singlegene(self, intensity_mutation, p_add_gene): 
        #Definimos la función de mutación "Single-Gene" que modifica un solo gen seleccionado de manera aleatoria entre sus genes.
        #Además, se selecciona de manera aleatoria qué caracteristica del gen será mutada (diametro, posición o color).
        #Finalmente se evalua si se agregará o no un nuevo gen al cromosoma.
        mutated_chromosome = deepcopy(self.chromosome)
        mutation_type = random.choice(mutated_chromosome[0].params)
        mutGene = random.randrange(0,len(mutated_chromosome))
        x_max = self.chromosome[0].size[0]
        y_max = self.chromosome[0].size[1]
        
        #Si la mutación se dará en el diámetro.
        if mutation_type == "diameter":
            diameter = int(mutated_chromosome[mutGene].diameter + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[mutGene].diameter)
            mutated_chromosome[mutGene].diameter = diameter
        #Si la mutación se dará en la posición.
        elif mutation_type == "position":
            x = int(mutated_chromosome[mutGene].position.x + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[mutGene].position.x)
            y = int(mutated_chromosome[mutGene].position.y + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[mutGene].position.y)
            mutated_chromosome[mutGene].position = Centroid(min(x,x_max),min(y,y_max))
        #Si la mutación se dará en el color.
        elif mutation_type == "color":
            r = int(mutated_chromosome[mutGene].color.r + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[mutGene].color.r)
            g = int(mutated_chromosome[mutGene].color.g + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[mutGene].color.g)
            b = int(mutated_chromosome[mutGene].color.b + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[mutGene].color.b)
            mutated_chromosome[mutGene].color = RGB_Color(min(r,255),min(g,255),min(b,255))
            
        #Se evalua si se agregará un nuevo gen al final del cromosoma.
        if random.random() < p_add_gene:
            mutated_chromosome.append(Gene((200,200)))
            
        return Chromosome(mutated_chromosome)    
        
    def mutate_allgenes(self, intensity_mutation, p_add_gene):
        #Definimos la función de mutación "All-Gene" que modifica todos los genes del cromosoma.
        #Además, se selecciona de manera aleatoria qué caracteristica del gen será mutada (diametro, posición o color).
        #Finalmente se evalua si se agregará o no un nuevo gen al cromosoma.
        mutated_chromosome = deepcopy(self.chromosome)
        mutation_type = random.choice(mutated_chromosome[0].params)
        x_max = self.chromosome[0].size[0]
        y_max = self.chromosome[0].size[1]
        
        for i in range(len(mutated_chromosome)):  
            #Si la mutación se dará en el diámetro.
            if mutation_type == "diameter":
                mutated_chromosome[i].diameter = (mutated_chromosome[i].diameter + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[i].diameter)
            #Si la mutación se dará en el diámetro.
            elif mutation_type == "position":            
                x = int(mutated_chromosome[i].position.x + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[i].position.x)
                y = int(mutated_chromosome[i].position.y + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[i].position.y)
                mutated_chromosome[i].position = Centroid(min(x,x_max),min(y,y_max))
            #Si la mutación se dará en el color.
            elif mutation_type == "color":
                r = int(mutated_chromosome[i].color.r + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[i].color.r)
                g = int(mutated_chromosome[i].color.g + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[i].color.g)
                b = int(mutated_chromosome[i].color.b + random.uniform(-1,1) * intensity_mutation * mutated_chromosome[i].color.b)
                mutated_chromosome[i].color = RGB_Color(min(r,255),min(g,255),min(b,255))
        
        #Se evalua si se agregará un nuevo gen al final del cromosoma.
        if random.random() < p_add_gene:
            mutated_chromosome.append(Gene((200,200)))

        return Chromosome(mutated_chromosome)

#--------------------------------------3.Cálculo de Fitness-----------------------------------------#

def calculate_fitness(img_reference, chromosome):  
    #Definimos la función para el cálculo del fitness.
    #Imagen de referencia:
    img_1 = np.array(img_reference,np.int16)
    #Generación de imagen a partir de los genes del cromosoma:
    img_2 = np.array(chromosome.Paint_Image(),np.int16)
    #Cálculo del valor absoluto de la diferencia entre ambas imágenes.
    diference_img = np.sum(np.abs(img_1-img_2))
    #Calculo del fitness...
    fitness = (1 / (diference_img + 1))*100
    
    return fitness

#------------------------------4.Inicialización de Población de Cromosoma----------------------------#

def init_population(popsize, chromosome_size, canvas_size):
    #Inicializamos la población de individuos y la estructura de sus genes iniciales.
    population = []
    for i in range(popsize):
        new_chromosome = [Gene((canvas_size[0],canvas_size[1])) for j in range(chromosome_size)]
        population.append(Chromosome(new_chromosome))
    return population

#--------------------------------5.Evaluación de Población de Cromosomas------------------------------#

def evaluate_population(population, img_reference):
    #Evaluamos el fitness de una población.
    for i in range(len(population)):
        if population[i].fitness == -1:
            population[i].fitness = calculate_fitness(img_reference, population[i])
    return population

#--------------------------------6.Operadores de Selección de Padres------------------------------#

# 6.1 Selección por Ruleta.
def select_parents_roulette(population):
    pop_size = len(population)
    ind_Parent1 = 0
    ind_Parent2 = 0
    # Escoje el primer padre:
    total_pop_fitness = sum([chromosome.fitness for chromosome in population])
    random_fitness = random.uniform(0, total_pop_fitness)
    sum_fitness = 0
    for i in range(pop_size):
        sum_fitness = sum_fitness + population[i].fitness
        if sum_fitness >= random_fitness: 
            ind_Parent1 = i
            break
     
    # Escoje el segundo padre, desconsiderando el padre ya escogido
    total_pop_fitness = total_pop_fitness - population[ind_Parent1].fitness # retira el fitness del padre ya escogido
    random_fitness = random.uniform(0, total_pop_fitness)   # escoge un numero aleatorio entre 0 y sumfitness
    sum_fitness = 0     # fitness acumulado
    for i in range(pop_size):
        if i == ind_Parent1: continue   # si es el primer padre 
        sum_fitness = sum_fitness + population[i].fitness
        if sum_fitness >= random_fitness: 
            ind_Parent2 = i
            break        
    return (population[ind_Parent1], population[ind_Parent2])

# 6.2 Selección por Torneo.
def select_parents_tournament(population,size_torneo):
    # Escoje el primer padre
    list_chromosomes=[]
    x1 = np.random.permutation(len(population))
    y1= x1[0:size_torneo]
    for i in range(size_torneo):
        list_chromosomes.append(population[y1[i]].fitness)
    max_ind = np.argmax(list_chromosomes)
    ind_parent1 = x1[max_ind]
    
    # Escoje el segundo padre, desconsiderando el primer padre   
    list_chromosomes=[]
    x2 = np.delete(x1, max_ind)
    x2 = np.random.permutation(x2)
    y2= x2[0:size_torneo]
    for i in range(size_torneo):
        list_chromosomes.append(population[y2[i]].fitness)
    max_ind=np.argmax(list_chromosomes)
    ind_parent2 = x2[max_ind]
    
    return (population[ind_parent1],population[ind_parent2])

#--------------------------------7.Operador de Selección de Sobrevivientes------------------------------#

# 7.1 Selección por Ranking.
def select_survivors_ranking(population, child_population, numsurvivors):
    new_population = []
    population.extend(child_population) # une las dos poblaciones
    isurvivors = sorted(range(len(population)), key=lambda i: population[i].fitness,reverse=True)[:numsurvivors]
    for i in range(numsurvivors): 
        new_population.append(population[isurvivors[i]])
    return new_population

#------------------------------------------8.Algoritmo Genético----------------------------------------#

def genetic_algorithm(population, img_reference, total_gen=100, p_mut=0.1, p_add_gene = 0.1, intensity_mut=0.5, 
                      crossover="onepoint", mutation="singlegene", 
                      selection_parents_method="roulette", 
                      selection_survivors_method="ranking"):
    
    pop_size = len(population)
    #Evaluamos la población inicial
    evaluate_population(population, img_reference)
    #Obtenemos el mejor individuo en base a la población evaluada previamente.
    ibest = sorted(range(len(population)), key=lambda i: population[i].fitness)[:1]
    #Obtenemos el mejor fitness en base a la población evaluada previamente.
    best_fitness = [population[ibest[0]].fitness]
    count = 0
    #Mostramos el fitness inicial con el que comienza a optimizar el AG.
    print("Fitness Inicial = {}".format(population[ibest[0]].fitness))
    
    for gen in range(total_gen):
        #Si se trabajaran con operadores de cruzamiento:
        if (crossover != "none"):
           #Seleccionamos las parejas de padres:
            mating_pool = []
            mating_pool_size = int(pop_size/2)
            if selection_parents_method=="roulette":
                for i in range(mating_pool_size): 
                    mating_pool.append(select_parents_roulette(population)) 
            elif selection_parents_method=="tournament":
                for i in range(mating_pool_size): 
                    mating_pool.append(select_parents_tournament(population,3)) 
            else:
                raise NotImplementedError

            #Creamos la siguiente población:
            child_population = []
            for i in range(len(mating_pool)):
                #Si el tipo de cruzamiento es OnePoint.
                if crossover == "onepoint":  
                    child_population.extend(mating_pool[i][0].crossover_onepoint(mating_pool[i][1]))
                #Si el tipo de cruzamiento es Uniform.
                elif crossover == "uniform": 
                    child_population.extend(mating_pool[i][0].crossover_uniform(mating_pool[i][1])) 
                #Si el tipo de cruzamiento es Aritmethic-Complete.
                elif crossover == "arithmetic_complete": 
                    child_population.extend(mating_pool[i][0].crossover_arithmetic(mating_pool[i][1])) 
                #Si el tipo de cruzamiento es Aritmethic-Unique.
                elif crossover == "arithmetic_unique": 
                    child_population.extend(mating_pool[i][0].crossover_arithmetic_unique(mating_pool[i][1]))
                else:
                    raise NotImplementedError

            #Aplicamos mutación sobre los hijos generados:
            for i in range(len(child_population)):
                if random.uniform(0, 1) < p_mut:
                    #Si el tipo de cruzamiento es Single-Gene.
                    if mutation == "singlegene":
                        child_population[i] = child_population[i].mutate_singlegene(intensity_mut, p_add_gene)
                    #Si el tipo de cruzamiento es All-Gene.
                    elif mutation == "allgenes":
                        child_population[i] = child_population[i].mutate_allgenes(intensity_mut, p_add_gene)
                    else:
                        raise NotImplementedError

            #Evaluamos la nueva población de hijos.
            evaluate_population(child_population, img_reference)  

            #Seleccionamos los sobrevivientes entre la nueva generación de los hijos y los padres utilizando "Ranking" como método.
            if selection_survivors_method == "ranking":
                population = select_survivors_ranking(population, child_population, pop_size)
            else:
                raise NotImplementedError
        
        #En caso NO se implementen operadores de cruzamiento:
        else:
            #Aplicamos mutación sobre los hijos generados:
            for i in range(len(population)):
                if random.uniform(0, 1) < p_mut:
                    #Si el tipo de cruzamiento es Single-Gene.
                    if mutation == "singlegene":
                        population[i] = population[i].mutate_singlegene(intensity_mut, p_add_gene)
                    #Si el tipo de cruzamiento es All-Gene.
                    elif mutation == "allgenes":
                        population[i] = population[i].mutate_allgenes(intensity_mut, p_add_gene)
                    else:
                        raise NotImplementedError

            #Evaluamos la nueva población de hijos.
            evaluate_population(population, img_reference)  

            #Seleccionamos los sobrevivientes con la población mutada usando "Ranking" como método.
            if selection_survivors_method == "ranking":
                population = select_survivors_ranking(population, [], pop_size)
            else:
                raise NotImplementedError
                
        #Obtenemos el mejor individuo en base a la población evaluada previamente.
        ibest = sorted(range(len(population)), key=lambda i: population[i].fitness)[:1]
        #Obtenemos el mejor fitness en base a la población evaluada previamente.
        best_fitness.append(population[ibest[0]].fitness)
        
        #Guardamos la imágen y el cromosoma del mejor individuo cada 100 generaciones.
        if (gen) % 100 == 0:
            file_save = open('population_ag.obj', 'wb')
            pickle.dump(population, file_save)
            directory = r'C:\Users\David\AG_Paint'
            os.chdir(directory) 
            name = "ag_paint_" + str(gen) + ".jpg"
            image = population[ibest[0]].Paint_Image()
            cv2.imwrite(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return population[ibest[0]], best_fitness, population[ibest[0]].chromosome

