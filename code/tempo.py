import pandas as pd 
import time
from functools import wraps
import matplotlib.pyplot as plt
import math
import networkx as nx


def condicional_prob2(graph,node_1,node_2):


    mapa = graph.groupby(0)[1].apply(set).to_dict()

    p_cima,p_baixo = [len(mapa.get(node_2,[]))/len(mapa)],[]

    vizinhança_node_1 = list(mapa.get(node_1,[]))
    p_baixo.append(len(mapa.get(vizinhança_node_1[1],[]))/len(mapa))

    i = 0

    condicionante_cima = mapa.get(node_2,set())
    condionate_baixo = mapa.get(vizinhança_node_1[1],set())


    while True:
        (mapa.get(vizinhança_node_1[i],set())).intersection(condicionante_cima)
        print((len(list((mapa.get(vizinhança_node_1[i],set()).intersection(condicionante_cima)))),len(list(condicionante_cima))))
        p_cima.append(len(list((mapa.get(vizinhança_node_1[i],set()).intersection(condicionante_cima))))/ len(list(condicionante_cima)))
        condicionante_cima = mapa.get(vizinhança_node_1[i],set()).intersection(condicionante_cima)
        i += 1 
        # print(len(condicionante_cima))
        if len(list(condicionante_cima)) <= 1:
            break
    i = 2

    # vizinhança_node_1 = list(mapa.get(node_1,[]))

    while True:
        (mapa.get(vizinhança_node_1[i],set())).intersection(condionate_baixo)
        p_baixo.append(len(list((mapa.get(vizinhança_node_1[i],set()).intersection(condionate_baixo))))/ len(list(condionate_baixo)))
        condionate_baixo = mapa.get(vizinhança_node_1[i],set()).intersection(condionate_baixo)
        i += 1 
        # print(len(condionate_baixo))
        if len(list(condionate_baixo))  <= 1 :
            break

    aux = 1
    for i in p_cima:
        aux = aux*i

    for j in p_baixo:
        aux = aux/j


    print(p_cima)
    print(p_baixo)
    return aux




def condicional_prob(graph,node):

    mapa = graph.groupby(0)[1].apply(set).to_dict()
    ps = [[],[]]
    # start with a list of keys (not a list containing a dict_keys object)
    indexes = list(mapa.keys())
    i = 0


    to_compare = list(mapa.get(node, []))
    if not to_compare:
        print("node {node} has no neighbors")
    else:
        while True:
            aux = 0
            new_indexes = []
            print(len(indexes))

            if i >= len(to_compare):
                break
            target = to_compare[i]

            for value in indexes:
                if target in mapa.get(value, set()):
                    new_indexes.append(value)
                    aux += 1

            if len(indexes) == 0:
                break
            ps[0].append(aux / len(indexes))

            i += 1
            indexes = new_indexes
            if len(indexes) <= 1:
                break

        i = 1


        to_compare = list(mapa.get(node, []))
        if not to_compare:
            print("node 107 has no neighbors")
        else:
            while True:
                aux = 0
                new_indexes = []

                if i >= len(to_compare):
                    break
                target = to_compare[i]

                for value in indexes:
                    if target in mapa.get(value, set()):
                        new_indexes.append(value)
                        aux += 1

                if len(indexes) == 0:
                    break
                ps[1].append(aux / len(indexes))
                print(len(indexes))
                i += 1
                indexes = new_indexes
                if len(indexes) <= 1:
                    break

        aux = 1
        for i in ps[0]:
            aux = aux*i

        for j in ps[1]:
            aux = aux/j

        return aux
    

def main():

    tempos = [[] for i in range(10)]
    # for i in range(14,15):
    #     print(i)
    #     G = nx.erdos_renyi_graph(2**i,0.2)
    #     df = pd.DataFrame(G.edges(),columns= None)
    grafo_original = pd.read_csv("/home/edu/Area_de_Trabalho/Projs/links/data/movies/testes/grafo_0.csv",header=None)
    grafo_invertido = grafo_original.rename(columns={0: 1, 1: 0})[[0,1]]

    # Concatena os dois
    grafo = pd.concat([grafo_original, grafo_invertido], ignore_index=True)
    print(condicional_prob2(grafo,107,1))

if __name__ == "__main__":
    main()