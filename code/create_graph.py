import pandas as pd 
from args import *
import numpy as np
import pandas as pd
import os
import pandas as pd

column_name_update = {
    "citeulike/users":      ["user_id",  "product_id"],
    "ml-100k":             ["userId",  "movieId"],  # ← sem /u.data
    "lastfm/user_artists": ["userID",  "artistID"],
    "amazon/Software" :        ["user_id", "parent_asin"]
}

dataset_format = {
    "citeulike/users":       ["\t",  ".dat"],
    "ml-100k":             ["\t", "data"],
    "lastfm/user_artists": ["\t", "dat"]
}

def create_target_set(bi_netw, testGraph, ratio=100):
    """
    Gera o conjunto de avaliação de forma ultra-rápida e segura para a memória,
    utilizando junções do Pandas em vez de conjuntos (sets) em Python puro.
    """
    
    col_left = bi_netw.columns[0]
    col_right = bi_netw.columns[1]
    
    left_nodes = bi_netw[col_left].dropna().unique().astype(np.int32)
    right_nodes = bi_netw[col_right].dropna().unique().astype(np.int32)
    
    test_positives = testGraph[[testGraph.columns[0], testGraph.columns[1]]].copy()
    test_positives.columns = [col_left, col_right]
    test_positives['label'] = 1
    
    num_negatives = len(test_positives) * ratio
    
    df_reais = bi_netw[[col_left, col_right]].copy()
    df_reais['real'] = 1
    
    
    negatives_list = []
    generated = 0
    batch_size = 2_000_000 
    rng = np.random.default_rng()
    len_left = len(left_nodes)
    len_right = len(right_nodes)
    
    while generated < num_negatives:
        idx_u = rng.integers(0, len_left, size=batch_size)
        idx_i = rng.integers(0, len_right, size=batch_size)
        
        df_batch = pd.DataFrame({
            col_left: left_nodes[idx_u], 
            col_right: right_nodes[idx_i]
        })
        
        df_batch = pd.merge(df_batch, df_reais, on=[col_left, col_right], how='left')
        df_batch = df_batch[df_batch['real'].isna()][[col_left, col_right]]
        df_batch = df_batch.drop_duplicates()
        
        needed = num_negatives - generated
        if len(df_batch) > needed:
            df_batch = df_batch.iloc[:needed]
            
        negatives_list.append(df_batch)
        generated += len(df_batch)
        
    neg_df = pd.concat(negatives_list, ignore_index=True)
    neg_df['label'] = 0
    
    target_df = pd.concat([test_positives, neg_df], ignore_index=True)
    target_df = target_df.sample(frac=1).reset_index(drop=True)
    
    del df_reais, negatives_list, df_batch
    import gc
    gc.collect()
    
    return target_df.astype(np.int32)

def create_and_save_target_set(bi_netw, testGraph, aval_path, ratio=100):
    """
    Gera as amostras negativas e escreve-as DIRETAMENTE no disco.
    Isto impede a formação de matrizes gigantes na RAM, anulando
    qualquer possibilidade de Segmentation Fault ou Double Free.
    """
    
    col_left = bi_netw.columns[0]
    col_right = bi_netw.columns[1]
    
    left_nodes = bi_netw[col_left].dropna().unique().astype(np.int32)
    right_nodes = bi_netw[col_right].dropna().unique().astype(np.int32)
    
    test_positives = testGraph[[testGraph.columns[0], testGraph.columns[1]]].copy()
    test_positives.to_csv(aval_path, index=False, header=False)
    
    num_negatives = len(test_positives) * ratio
    print(f"A gerar e a escrever {num_negatives} amostras negativas à medida que avança...")
    
    # Tabela auxiliar para filtragem rápida
    df_reais = bi_netw[[col_left, col_right]].copy()
    df_reais['real'] = 1
    
    generated = 0
    batch_size = 2_000_000
    
    rng = np.random.default_rng()
    len_left = len(left_nodes)
    len_right = len(right_nodes)
    
    with open(aval_path, 'a') as f:
        while generated < num_negatives:
            idx_u = rng.integers(0, len_left, size=batch_size)
            idx_i = rng.integers(0, len_right, size=batch_size)
            
            df_batch = pd.DataFrame({
                col_left: left_nodes[idx_u], 
                col_right: right_nodes[idx_i]
            })
            
            df_batch = pd.merge(df_batch, df_reais, on=[col_left, col_right], how='left')
            df_batch = df_batch[df_batch['real'].isna()][[col_left, col_right]]
            df_batch = df_batch.drop_duplicates()
            
            needed = num_negatives - generated
            if len(df_batch) > needed:
                df_batch = df_batch.iloc[:needed]
                
            if not df_batch.empty:
                df_batch.to_csv(f, index=False, header=False)
                generated += len(df_batch)
                
            del df_batch
            
    del df_reais
    import gc
    gc.collect()
    print("✅ Gravação concluída com consumo nulo de memória agregada!")


def create_graph(args):

    if args.dataset == "ml-100k":
        data = pd.read_csv(
            f"{args.exec_path}/data/ml-100k/u.data",
            sep="\t",
            header=None,
            names=["leftNodes", "rightNodes", "rating", "timestamp"]
        )
        data = data[["leftNodes", "rightNodes"]]

    

    elif args.dataset.startswith("lastfm"):
        filepath = f"{args.exec_path}/data/lastfm/user_artists.dat"
        print("A processar o Last.fm...")
        
        data = pd.read_csv(filepath, sep='\t', usecols=['userID', 'artistID'])
        data['leftNodes'] = pd.factorize(data['userID'])[0]
        data['rightNodes'] = pd.factorize(data['artistID'])[0]
        
        data = data[['leftNodes', 'rightNodes']]
        data.drop_duplicates(inplace=True)
        
    elif args.dataset.startswith("citeulike"):
        filepath = f"{args.exec_path}/data/citeulike/users.dat" 
        print("A processar o users.dat do CiteULike...")
        
        left_nodes = []
        right_nodes = []
        with open(filepath, 'r') as f:
            for user_id, line in enumerate(f):
                items = line.strip().split()
                if not items:
                    continue
                for item_str in items[1:]:
                    left_nodes.append(user_id)
                    right_nodes.append(int(item_str))
                    
        data = pd.DataFrame({"leftNodes": left_nodes, "rightNodes": right_nodes})
        data.drop_duplicates(inplace=True)

    else:
        raise ValueError(f"Dataset desconhecido: {args.dataset}")

    data["rightNodes"] = data["rightNodes"] + data.leftNodes.max() + 1
    data = data[["leftNodes", "rightNodes"]]


    print("Amostrando Subgrafo Induzido...")
    if len(data) > 50000:
        tecnica = str(args.SampleTecnic).lower().strip()
        print(f"Dataset grande ({len(data)} arestas). Aplicando amostragem: {tecnica.upper()}")
        
        if tecnica == "random":
            data = sample_random_subgraph(data, max_edges=25000, seed=args.CurrentSeed)
        elif tecnica == "degree":
            data = sample_degree_subgraph(data, max_edges=25000, seed=args.CurrentSeed)
        elif tecnica == "fire":
            data = sample_forest_fire(data, max_edges=25000, p=0.7, seed=args.CurrentSeed)
        else:
            print(f"⚠️ Método '{tecnica}' não reconhecido! Usando 'random' como padrão.")
            data = sample_random_subgraph(data, max_edges=50000, seed=args.CurrentSeed)
    else:
        print(f"Dataset pequeno ({len(data)} arestas). Mantendo o tamanho original.")

    trainGraph, testGraph = TrainTestSep(data, args)
    
    train_path = os.path.join(args.exec_path, "data", "exp", f"{args.dataset.split('/')[0]}", f"{args.SampleTecnic}", f"trainGraph{args.CurrentSeed}.csv")
    test_path  = os.path.join(args.exec_path, "data", "exp", f"{args.dataset.split('/')[0]}", f"{args.SampleTecnic}", f"testGraph{args.CurrentSeed}.csv")
    aval_path  = os.path.join(args.exec_path, "data", "exp", f"{args.dataset.split('/')[0]}", f"{args.SampleTecnic}", f"targetGraph{args.CurrentSeed}_{args.Filter}.csv")
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    trainGraph.to_csv(train_path, index=False, header=False)
    testGraph.to_csv(test_path, index=False, header=False)
    
    

    ratio_seguro = 100

    create_and_save_target_set(data, testGraph, aval_path, ratio=ratio_seguro)
 
    
def filter_k_core(df, k=20):
    """ Remove nós com grau menor que k de forma iterativa """
    while True:
        # Contar graus de ambos os lados
        left_counts = df['leftNodes'].value_counts()
        right_counts = df['rightNodes'].value_counts()
        
        # Identificar nós que mantêm o critério
        keep_left = left_counts[left_counts >= k].index
        keep_right = right_counts[right_counts >= k].index
        
        before = len(df)
        df = df[df['leftNodes'].isin(keep_left) & df['rightNodes'].isin(keep_right)]
        after = len(df)
        
        # Se não houver mais nada para remover, para
        if before == after:
            break
    return df

def sample_node_induced_subgraph(edges_df, max_edges=10000, seed=42):
    rng = np.random.default_rng(seed)
    
    # Se já for menor que o limite, não faz nada
    if len(edges_df) <= max_edges:
        return edges_df

    # Estimativa de ratio para atingir as edges desejadas (raiz quadrada)
    # Como as edges caem quadraticamente, a proporção de nós é a raiz da proporção de edges
    current_edges = len(edges_df)
    ratio = (max_edges / current_edges) ** 0.5
    
    all_left = edges_df['leftNodes'].unique()
    all_right = edges_df['rightNodes'].unique()
    
    sampled_left = rng.choice(all_left, size=int(len(all_left) * ratio), replace=False)
    sampled_right = rng.choice(all_right, size=int(len(all_right) * ratio), replace=False)
    
    subgraph = edges_df[
        edges_df['leftNodes'].isin(sampled_left) & 
        edges_df['rightNodes'].isin(sampled_right)
    ].copy()
    
    return subgraph

def TrainTestSep(graph, args):
    print("A separar Treino e Teste de forma Vetorizada (Anti-Crash)...")
    
    # 1. Amostragem Super-Rápida em C-level
    # Agrupa por utilizador (leftNodes) e tira 20% das interações para o Teste
    # NOTA: Se na sua tese usava 1 item fixo em vez de 20%, 
    # troque `frac=0.2` por `n=1`
    try:
        graphTest = graph.groupby("leftNodes", group_keys=False).sample(frac=0.2, random_state=args.CurrentSeed)
    except Exception as e:
        # Fallback de segurança se algum nó tiver comportamentos anómalos
        print(f"Aviso no agrupamento: {e}. A usar separação global...")
        graphTest = graph.sample(frac=0.2, random_state=args.CurrentSeed)
    
    # 2. O Treino será tudo o que não foi selecionado para Teste
    graphTrain = graph.drop(graphTest.index)
    
    # 3. Limpeza e reordenação de índices
    graphTrain = graphTrain.reset_index(drop=True)
    graphTest = graphTest.reset_index(drop=True)
    
    # 4. Validação estrita para evitar Vazamento de Dados (Data Leakage)
    overlap = graphTrain.merge(graphTest, on=["leftNodes","rightNodes"])
    
    if len(overlap) > 0:
        print(f"ALERTA CRÍTICO: {len(overlap)} arestas vazaram para o teste!")
    else:
        print("✅ Separação perfeita: 0 interseções entre Treino e Teste.")
        
    return graphTrain, graphTest


def sample_random_subgraph(edges_df, max_edges=50000, seed=42):
    """ 1. RANDOM (O método atual): Amostragem uniforme de nós induzindo o subgrafo. """
    rng = np.random.default_rng(seed)
    if len(edges_df) <= max_edges:
        return edges_df

    ratio = (max_edges / len(edges_df)) ** 0.5
    all_left = edges_df['leftNodes'].unique()
    all_right = edges_df['rightNodes'].unique()
    
    sampled_left = rng.choice(all_left, size=int(len(all_left) * ratio), replace=False)
    sampled_right = rng.choice(all_right, size=int(len(all_right) * ratio), replace=False)
    
    subgraph = edges_df[
        edges_df['leftNodes'].isin(sampled_left) & 
        edges_df['rightNodes'].isin(sampled_right)
    ].copy()
    
    return subgraph

def sample_degree_subgraph(edges_df, max_edges=50000, seed=42):
    """ 2. DEGREE-BASED: Nós com mais conexões (Hubs) têm maior probabilidade de serem amostrados. """
    rng = np.random.default_rng(seed)
    if len(edges_df) <= max_edges:
        return edges_df

    # Conta o grau de cada nó
    left_counts = edges_df['leftNodes'].value_counts()
    right_counts = edges_df['rightNodes'].value_counts()
    
    # Transforma a contagem em probabilidades (Normalização)
    left_probs = left_counts / left_counts.sum()
    right_probs = right_counts / right_counts.sum()
    
    # A raiz quadrada compensa o facto de estarmos a amostrar os dois lados da matriz
    ratio = (max_edges / len(edges_df)) ** 0.5
    
    # Amostra os nós usando a probabilidade (p) baseada no grau
    sampled_left = rng.choice(left_counts.index, size=int(len(left_counts) * ratio), replace=False, p=left_probs.values)
    sampled_right = rng.choice(right_counts.index, size=int(len(right_counts) * ratio), replace=False, p=right_probs.values)
    
    subgraph = edges_df[
        edges_df['leftNodes'].isin(sampled_left) & 
        edges_df['rightNodes'].isin(sampled_right)
    ].copy()
    
    return subgraph

def sample_forest_fire(edges_df, max_edges=50000, p=0.7, seed=42):
    """ 3. FOREST FIRE: Simula um 'incêndio' a partir de nós semente. Preserva comunidades e clusters. """
    rng = np.random.default_rng(seed)
    if len(edges_df) <= max_edges:
        return edges_df

    print("   -> Construindo Adjacência Bidirecional para o Forest Fire...")
    # Constrói adjacência bidirecional rápida
    adj = {}
    left_vals = edges_df['leftNodes'].values
    right_vals = edges_df['rightNodes'].values
    
    for u, v in zip(left_vals, right_vals):
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    visited_nodes = set()
    nodes = list(adj.keys())
    rng.shuffle(nodes) # Randomiza a ordem das sementes
    
    edges_coletadas = 0
    
    print("   -> Iniciando propagação do fogo...")
    for start_node in nodes:
        if edges_coletadas >= max_edges:
            break
        if start_node in visited_nodes:
            continue
            
        queue = [start_node]
        visited_nodes.add(start_node)
        
        while queue and edges_coletadas < max_edges:
            u = queue.pop(0)
            neighbors = adj[u]
            
            unvisited = [v for v in neighbors if v not in visited_nodes]
            if not unvisited: continue
            
            # O "fogo" espalha-se geometricamente, ou mantemos uma fração p dos vizinhos
            burn_count = rng.geometric(1.0 - p) if p < 1.0 else len(unvisited)
            burn_count = min(burn_count, len(unvisited))
            
            if burn_count > 0:
                burned = rng.choice(unvisited, size=burn_count, replace=False)
                for v in burned:
                    visited_nodes.add(v)
                    queue.append(v)
                    edges_coletadas += 1 # Estimativa de arestas capturadas
                    
    # Induzir o subgrafo baseado nos nós que foram "queimados"
    subgraph = edges_df[
        edges_df['leftNodes'].isin(visited_nodes) & 
        edges_df['rightNodes'].isin(visited_nodes)
    ].copy()
    
    # Se o fogo se descontrolou e passou o limite, cortamos o excesso aleatoriamente
    if len(subgraph) > max_edges:
        subgraph = subgraph.sample(n=max_edges, random_state=seed)
    return subgraph