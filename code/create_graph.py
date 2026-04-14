import pandas as pd 
from args import *
import networkx as nx
import random 
import numpy as np
import math
import pandas as pd
from collections import deque
import os
from scipy.io import mmread
import pandas as pd


column_name_update = {
    "movies/rating":       ["userId",  "movieId"],
    "ml-100k":             ["userId",  "movieId"],  # ← sem /u.data
    "lastfm/user_artists": ["userID",  "artistID"]
}

dataset_format = {
    "movies/rating":       [",",  "csv"],
    "ml-100k":             ["\t", "data"],
    "lastfm/user_artists": ["\t", "dat"]
}
def read_mtx_file(file_path):

    # carregar matriz
    mat = mmread(file_path)

    # converter para COO (formato com índices)
    mat = mat.tocoo()

    # criar DataFrame de arestas
    df = pd.DataFrame({
        "leftNodes": mat.row,
        "rightNodes": mat.col
    })

    return df



def create_graph_normal_degree(data,args):
    left_deg = data.groupby("leftNodes").size()
    right_deg = data.groupby("rightNodes").size()

    left_nodes = np.random.choice(
        left_deg.index,
        size=args.GraphSize,
        replace=False,
        p=left_deg / left_deg.sum()
    )

    right_nodes = np.random.choice(
        right_deg.index,
        size=args.GraphSize*2,
        replace=False,
        p=right_deg / right_deg.sum()
    )

    return data[
        (data.leftNodes.isin(left_nodes)) &
        (data.rightNodes.isin(right_nodes))
    ]


def create_graph(args):


    print(args.dataset)
    if args.dataset == "amazon":
        data = read_mtx_file(f"{args.exec_path}/data/{args.dataset}/com-Amazon.mtx")

        # data["rightNodes"] = data["rightNodes"] + data.leftNodes.max()
    elif args.dataset == "ml-100k":
        data = pd.read_csv(
            f"{args.exec_path}/data/ml-100k/u.data",
            sep="\t",
            header=None,
            names=["leftNodes", "rightNodes", "rating", "timestamp"]
        )
        data = data[["leftNodes", "rightNodes"]]
    else:
        data = pd.read_csv(
            f"{args.exec_path}/data/{args.dataset}.{dataset_format[args.dataset][1]}",
            sep=dataset_format[args.dataset][0]
        )
        data.rename(columns={
            column_name_update[args.dataset][0]: "leftNodes",
            column_name_update[args.dataset][1]: "rightNodes"
        }, inplace=True)

    data["rightNodes"] = data["rightNodes"] + data.leftNodes.max()
    data = data[["leftNodes","rightNodes"]]

    if args.SampleTecnic == "normal":
        bi_netw = create_graph_normal(data, args)
    elif args.SampleTecnic == "degree":
        bi_netw = create_graph_normal_degree(data, args)
    elif args.SampleTecnic == "topology":
        bi_netw = create_forest_fire_graph(data, args)
    else:
        raise ValueError(f"SampleTecnic desconhecido: {args.SampleTecnic}")

    trainGraph, testGraph = TrainTestSep(bi_netw,args)
    save_graphs(trainGraph,testGraph,args)

import pandas as pd
import numpy as np
import random
from collections import deque

def create_forest_fire_graph(data, args):
    # 1. Construção das adjacências
    adj_left = {}
    adj_right = {}
    for l, r in zip(data["leftNodes"], data["rightNodes"]):
        adj_left.setdefault(l, set()).add(r)
        adj_right.setdefault(r, set()).add(l)

    # Parâmetros
    target_left = int(args.GraphSize)
    target_right = int(args.GraphSize)
    p = getattr(args, 'p_fire', 0.3) # Probabilidade de queima (ajuste conforme necessário)
    
    np.random.seed(args.RandomSeed)
    random.seed(args.RandomSeed)

    visited_left = set()
    visited_right = set()
    queue = deque()

    # Loop principal para garantir que atingimos o tamanho mesmo se o "fogo" apagar
    while len(visited_left) < target_left or len(visited_right) < target_right:
        # Se a fila esvaziar, escolhemos um novo nó aleatório (nova faísca)
        remaining_left = list(set(adj_left.keys()) - visited_left)
        if not remaining_left: break # Não há mais nós para explorar
        
        start_node = random.choice(remaining_left)
        if len(visited_left) < target_left:
            visited_left.add(start_node)
            queue.append((start_node, "left"))

        while queue:
            curr_node, side = queue.popleft()

            # Define vizinhos e limites baseado no lado atual
            if side == "left":
                neighbors = list(adj_left.get(curr_node, set()) - visited_right)
                target_set, other_side, limit = visited_right, "right", target_right
            else:
                neighbors = list(adj_right.get(curr_node, set()) - visited_left)
                target_set, other_side, limit = visited_left, "left", target_left

            if not neighbors:
                continue

            # Geometrically distributed number of neighbors to burn: 
            # Na prática, simulamos selecionando uma fração p dos vizinhos
            num_to_burn = np.random.geometric(1 - p) if p < 1.0 else len(neighbors)
            burned_neighbors = random.sample(neighbors, min(len(neighbors), num_to_burn))

            for n in burned_neighbors:
                if len(target_set) < limit:
                    target_set.add(n)
                    queue.append((n, other_side))
                
            if len(visited_left) >= target_left and len(visited_right) >= target_right:
                break

    # Filtrar edges
    df_sub = data[
        data["leftNodes"].isin(visited_left) &
        data["rightNodes"].isin(visited_right)
    ].copy().reset_index(drop=True)
    
    return df_sub



def create_graph_normal(data,args):
    # left_nodes = np.random.choice(data.leftNodes.unique(),1450, replace=False)
    # sub = data[data.leftNodes.isin(left_nodes)]
    # print(sub)

    return data[["leftNodes","rightNodes"]]




def create_snowball_graph(data, args):
    # 1. Construção das listas de adjacência (Bipartido)
    adj_left = {}
    adj_right = {}
    for l, r in zip(data["leftNodes"], data["rightNodes"]):
        adj_left.setdefault(l, set()).add(r)
        adj_right.setdefault(r, set()).add(l)

    # Configurações de alvos
    target_left = int(args.GraphSize)
    target_right = int(args.GraphSize)
    
    np.random.seed(args.RandomSeed)
    random.seed(args.RandomSeed)

    # 2. Inicialização
    all_left_nodes = list(adj_left.keys())
    start_node = random.choice(all_left_nodes)

    visited_left = {start_node}
    visited_right = set()
    queue = deque([start_node])

    # 3. Processo Snowball (BFS)
    # Continua enquanto houver fila e não atingirmos os dois limites
    while queue and (len(visited_left) < target_left or len(visited_right) < target_right):
        curr_l = queue.popleft()

        # Explora vizinhos à direita (Right Nodes)
        if curr_l in adj_left:
            for r in adj_left[curr_l]:
                if r not in visited_right:
                    if len(visited_right) < target_right:
                        visited_right.add(r)
                        
                        # Explora vizinhos de volta para a esquerda (Left Nodes)
                        if r in adj_right:
                            for next_l in adj_right[r]:
                                if next_l not in visited_left:
                                    if len(visited_left) < target_left:
                                        visited_left.add(next_l)
                                        queue.append(next_l)
                                    else:
                                        break # Atingiu limite de Left
                
                if len(visited_right) >= target_right and len(visited_left) >= target_left:
                    break

    # 4. Filtragem do DataFrame original
    # Mantemos apenas as arestas onde AMBOS os nós foram visitados
    df_sub = data[
        data["leftNodes"].isin(visited_left) & 
        data["rightNodes"].isin(visited_right)
    ].copy().reset_index(drop=True)

    return df_sub
def TrainTestSep(graph,args):
    DegreeCounter = graph.groupby("leftNodes").size()
    eligible = DegreeCounter[DegreeCounter >= 20].index
    graph = graph[graph["leftNodes"].isin(eligible)]

    DegreeCounter = DegreeCounter[eligible]






    sampled_parts = []
    for left_node, deg in DegreeCounter.items():
        n = int(math.ceil(deg * args.perct))
        part = graph[graph["leftNodes"] == left_node].sample(n=n, random_state=42)
        sampled_parts.append(part)

    if sampled_parts:
        graphTest = pd.concat(sampled_parts, ignore_index=True)
    else:
        graphTest = graph.iloc[0:0]  # empty dataframe with same columns

    graphTrain = graph.merge(graphTest, how='left', indicator=True)    
    graphTrain = graphTrain[graphTrain['_merge'] == 'left_only'].drop(columns='_merge')
    
    # print("Grau treino:", graphTrain.groupby("leftNodes").size().describe())
    # print("Grau teste:", graphTest.groupby("leftNodes").size().describe())

    # Razão esperada: grau_teste / grau_total ≈ perct (0.1)
    ratio = graphTrain.groupby("leftNodes").size() / DegreeCounter
    
    # print("Razão real teste/total:", ratio.describe())
    # # Checar duplicatas
    # print("Duplicatas no grafo:", graph.duplicated().sum())

    # Checar overlap após split
    overlap = graphTrain.merge(graphTest, on=["leftNodes","rightNodes"])
    # print("Overlap treino/teste:", len(overlap))
    return graphTrain,graphTest


import numpy as np
import pandas as pd

def topology_aware_sampling(data, args):
    # graus
    left_degree = data.groupby("leftNodes").size()
    right_degree = data.groupby("rightNodes").size()

    # probabilidades proporcionais ao grau
    p_left = left_degree / left_degree.sum()
    p_right = right_degree / right_degree.sum()

    # sample mantendo tamanho
    np.random.seed(args.RandomSeed)

    sampled_left = np.random.choice(
        left_degree.index,
        size=args.GraphSize,
        replace=False,
        p=p_left.values
    )

    sampled_right = np.random.choice(
        right_degree.index,
        size=args.GraphSize,
        replace=False,
        p=p_right.values
    )

    # filtra edges
    df_sub = data[
        data["leftNodes"].isin(sampled_left) &
        data["rightNodes"].isin(sampled_right)
    ].reset_index(drop=True)

    return df_sub[["leftNodes","rightNodes"]]


def apply_random_perturbation(graph):
    """
    Remove uma aresta aleatória e adiciona uma aresta entre
    um nó do lado esquerdo e um do lado direito.
    """

    try:
        left_set, right_set = nx.bipartite.sets(graph)
    except Exception:
        left_set = {n for n,d in graph.nodes(data=True) if d.get("bipartite") == 0}
        right_set = {n for n,d in graph.nodes(data=True) if d.get("bipartite") == 1}



    left = list(left_set)
    right = list(right_set)
    edges = graph.edges()

    #removing random edges 
    edges_df = pd.DataFrame(edges,columns=["leftNodes","rightNodes"])
    edges_to_delete = edges_df.groupby("leftNodes").apply(lambda x : x.sample(n=3)).reset_index(drop=True)


    graph.edge_subgraph(edges_to_delete.itertuples(index=False, name=None)).copy()

    #appending random edges
    rng = np.random.default_rng()
    rand = rng.integers(low=0,high=len(right)-1,size=int(len(left)*3))

    
    append_edges = [(left[i%len(left)],right[rand[i]]) for i in range(len(left)*3)]

    graph.add_edges_from(append_edges)    


    return graph


def save_graphs(trainGraph,testGraph,args):
    train_path = os.path.join(args.exec_path, "data", "exp",f"{args.dataset.split('/')[0]}",f"{args.SampleTecnic}","trainGraph.csv")
    test_path  = os.path.join(args.exec_path, "data", "exp",f"{args.dataset.split('/')[0]}",f"{args.SampleTecnic}","testGraph.csv")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    trainGraph.to_csv(train_path, header=False, index=False)
    testGraph.to_csv(test_path, header=False, index=False)


