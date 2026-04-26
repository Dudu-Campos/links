import pandas as pd

import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from scipy.stats import entropy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque, defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools
import gc
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import time
import math
import scipy.sparse as sp
import numpy as np
import traceback
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import gc
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import collections
from sklearn.metrics import average_precision_score, roc_auc_score
import subprocess
import collections 
import gc
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import os
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import average_precision_score
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import traceback
import sys
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler



dtype = np.dtype([
    ("col", np.int32),
    ("row1", np.int32),
    ("prob", np.float32)
])

def evaluate_our_method(X_train, y_train, X_test, y_test, device='cuda', epochs=100, batch_size=256):
    """
    Treina e avalia o modelo preditivo usando as features topológicas artesanais 
    (Entropia, Curtose, Assimetria, Histogramas, Graus).
    """
    print("="*50)
    print("INICIANDO AVALIAÇÃO: NOSSO MÉTODO (SMALL NET)")
    print("="*50)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo selecionado: {device.upper()}")
    results_our_method = {"aupr": None, "auroc": None, "status": "Pending"}
    
    try:
        # 1. Transferência para Dispositivo (CPU/GPU)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

        input_dim = X_train.shape[1]
        model = SmallNet(input_dim).to(device)
        
        # 2. Configuração de Perda e Otimização
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

        # 3. Loop de Treinamento
        print(f"Treinando MLP com {input_dim} features de entrada por {epochs} épocas...")
        model.train()
        dataset_size = X_train_tensor.size(0)
        
        for epoch in range(epochs):
            # Embaralha os índices para os mini-batches
            permutation = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 4. Fase de Inferência
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            # Aplica Sigmoid para converter logits em probabilidades (0 a 1)
            test_probs = torch.sigmoid(test_outputs).cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()

        # 5. Avaliação de Métricas
        aupr = average_precision_score(y_test_np, test_probs)
        auroc = roc_auc_score(y_test_np, test_probs)
        
        results_our_method["aupr"] = aupr
        results_our_method["auroc"] = auroc
        results_our_method["status"] = "Success"
        
        print(f"[Nosso Método] Sucesso! AUPR: {aupr:.4f} | AUROC: {auroc:.4f}")

    except torch.cuda.OutOfMemoryError as e:
        print("[Nosso Método] ERRO FATAL: Estouro de Memória de Vídeo (OOM).")
        results_our_method["status"] = "Failed (OOM)"
        
    except MemoryError as e:
        print("[Nosso Método] ERRO FATAL: Estouro de Memória RAM.")
        results_our_method["status"] = "Failed (RAM)"
        
    except Exception as e:
        print(f"[Nosso Método] ERRO INESPERADO: {type(e).__name__}")
        print(f"Mensagem: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        results_our_method["status"] = f"Failed ({type(e).__name__})"
        
    finally:
        # 6. Limpeza de Memória Rigorosa
        print("[Nosso Método] Limpando cache e liberando recursos...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    print("\n" + "="*50)
    print("RESUMO DA AVALIAÇÃO DO NOSSO MÉTODO:")
    if results_our_method["status"] == "Success":
        print(f" -> AUPR = {results_our_method['aupr']:.4f}")
    else:
        print(f" -> Status: {results_our_method['status']}")
    print("="*50)

    return results_our_method

class SmallNet(nn.Module):
    def __init__(self, input_dim):
        super(SmallNet, self).__init__()
        # Uma arquitetura "funil" simples e rápida
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1) # Saída única (probabilidade do link existir)
        )

    def forward(self, x):
        return self.network(x)
 
class WideAndDeepNet(nn.Module):
    def __init__(self, input_dim):
        super(WideAndDeepNet, self).__init__()
        
        # A parte "Deep" (Mistura tudo para achar padrões complexos não-lineares)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # A camada final "Wide" recebe a saída profunda (32) + as features originais puras (input_dim)
        self.final = nn.Linear(32 + input_dim, 1)

    def forward(self, x):
        # Passa os dados pela rede profunda
        deep_features = self.deep(x)
        
        # Junta (concatena) a saída profunda com a entrada original intacta
        combined = torch.cat([deep_features, x], dim=1) 
        
        # Faz a predição final
        return self.final(combined)


# 2. O SEU WRAPPER ATUALIZADO
class SklearnMLPWrapper:
    def __init__(self, epochs=100, batch_size=512):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model = None
        # Instanciamos o Scaler
        self.scaler = StandardScaler() 

    def fit(self, X, y):
        # Treina o Scaler e transforma o X_train
        X_scaled = self.scaler.fit_transform(X)
        
        # Converte para tensores usando os dados normalizados
        X_t = torch.tensor(np.array(X_scaled), dtype=torch.float32).to(self.device)
        y_t = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # ---> AQUI ESTÁ A MUDANÇA: Instancia o novo modelo WideAndDeepNet <---
        self.model = WideAndDeepNet(X.shape[1]).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
        self.model.train()
        dataset_size = X_t.size(0)
        for epoch in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for i in range(0, dataset_size, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(X_t[batch_idx]), y_t[batch_idx])
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        
        # Transforma o X_test usando o Scaler já treinado (NÃO usamos fit_transform aqui!)
        X_scaled = self.scaler.transform(X)
        
        X_t = torch.tensor(np.array(X_scaled), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = torch.sigmoid(self.model(X_t)).cpu().numpy()
        
        # Prepara a saída no formato Scikit-Learn [prob_falso, prob_verdadeiro]
        out = np.zeros((preds.shape[0], 2))
        out[:, 1] = preds[:, 0]        
        out[:, 0] = 1.0 - preds[:, 0]  
        return out
    
def build_sets(train_path, test_path):

    # CORREÇÃO: Forçar tudo para inteiro
    train = pd.read_csv(train_path, header=None).astype(int)
    test  = pd.read_csv(test_path, header=None).astype(int)

    train_set = set(zip(train[0], train[1]))
    test_set  = set(zip(test[0], test[1]))
    return train_set, test_set


def readData(dir, reverse=False):
    all_data = []
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if file_name.endswith(".bin"):
            data = pd.DataFrame(np.fromfile(file_path, dtype=dtype))
            all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data["col"]  = combined_data["col"].astype(int)
    combined_data["row1"] = combined_data["row1"].astype(int)

    # index com todos os pares
    dfProbsIndex = combined_data[["col", "row1"]].copy()
    dfProbsIndex = dfProbsIndex.rename(columns={"col": "leftNode", "row1": "rightNode"})
    dfProbsIndex = dfProbsIndex.drop_duplicates()

    if reverse:
        dfProbsIndex[["leftNode", "rightNode"]] = dfProbsIndex[["rightNode", "leftNode"]]

    # probs brutas — pivot por (leftNode, rightNode)
    combined_data["ck_idx"] = combined_data.groupby(
        ["col", "row1"]).cumcount()

    dfProbsValues = combined_data.pivot_table(
        index=["col", "row1"],
        columns="ck_idx",
        values="prob",
        aggfunc="first"
    ).reset_index()

    dfProbsValues.columns = ["leftNode", "rightNode"] + \
                            [f"prob_{i}" for i in range(dfProbsValues.shape[1] - 2)]
    dfProbsValues = dfProbsValues.fillna(0)

    # labels globais
    train_path = os.path.join(os.path.dirname(dir), "trainGraph.csv")
    test_path  = os.path.join(os.path.dirname(dir), "testGraph.csv")

    dfExcluded    = readFormatExcludedlinks(test_path)
    OriginalEdges = pd.read_csv(train_path, header=None)\
                      .rename(columns={0: "leftNode", 1: "rightNode"})\
                      .astype(int)

    excl_set  = set(zip(dfExcluded["leftNode"].astype(int),
                        dfExcluded["rightNode"].astype(int)))
    train_set = set(zip(OriginalEdges["leftNode"],
                        OriginalEdges["rightNode"]))

    dfProbsIndex["label"]    = [
        1 if (l, r) in excl_set  else 0
        for l, r in zip(dfProbsIndex["leftNode"], dfProbsIndex["rightNode"])
    ]
    dfProbsIndex["original"] = [
        1 if (l, r) in train_set else 0
        for l, r in zip(dfProbsIndex["leftNode"], dfProbsIndex["rightNode"])
    ]

    return dfProbsIndex, dfProbsValues







def getL3(args, train_graph_path):
    train_graph_df = pd.read_csv(train_graph_path, header=None)
    train_graph_df[[0,1]] = train_graph_df[[1,0]]
    train_graph_df["weight"] = 1
    if os.path.exists(f"{args.exec_path}/data/kpisti-L3-a163e9f/graph.txt"): os.remove(f"{args.exec_path}/data/kpisti-L3-a163e9f/graph.txt")
    train_graph_df.to_csv(f"{args.exec_path}/data/kpisti-L3-a163e9f/graph.txt", header=None, sep=" ", index=False)
    inicio = time.perf_counter()
    subprocess.run(f"{args.exec_path}/data/kpisti-L3-a163e9f/teste.out {args.exec_path}/data/kpisti-L3-a163e9f/graph.txt", shell=True)
    subprocess.run(f"mv {args.exec_path}/data/kpisti-L3-a163e9f/L3_predictions_graph.txt.dat {args.exec_path}/data/kpisti-L3-a163e9f/L3_predictions_graph.txt", shell=True)
    fim = time.perf_counter()
    total_time = fim - inicio
    result = pd.read_csv(f"{args.exec_path}/data/kpisti-L3-a163e9f/L3_predictions_graph.txt", sep="\t", header=None)
    result.columns = ["rightNode","leftNode", "pred"]
    result["leftNode"] = result["leftNode"].astype(int)
    result["rightNode"] = result["rightNode"].astype(int)
 
    return [result[["leftNode","rightNode", "pred"]],total_time]

def getCommonNeighbors(train_graph_path):
    """
    Compute a bipartite-aware "common neighbors" score for left->right pairs.
    We use the biadjacency matrix B (L x R) and compute M = B * B.T * B (L x R).
    M[i,j] counts length-3 paths left_i -> right_k -> left_l -> right_j, a useful
    bipartite analogue for common-neighbors-based link scoring.
    Returns a DataFrame with columns ["rightNode","leftNode","pred"] sorted by pred desc.
    """

    g = pd.read_csv(train_graph_path, header=None)
    if g.shape[1] < 2:
        raise ValueError("train_graph must have at least two columns (left,right)")
    g = g.rename(columns={0: "leftNode", 1: "rightNode"})
    g["leftNode"] = g["leftNode"].astype(int)
    g["rightNode"] = g["rightNode"].astype(int)

    left_nodes = pd.Index(g["leftNode"].unique())
    right_nodes = pd.Index(g["rightNode"].unique())
    nL = len(left_nodes)
    nR = len(right_nodes)
    idxL = {n: i for i, n in enumerate(left_nodes)}
    idxR = {n: i for i, n in enumerate(right_nodes)}

    rows = g["leftNode"].map(idxL).values
    cols = g["rightNode"].map(idxR).values
    data = np.ones(len(rows), dtype=float)
    inicio = time.time()

    B = sp.csr_matrix((data, (rows, cols)), shape=(nL, nR), dtype=float)

    M = (B.dot(B.T)).dot(B)  # (LxL) * (LxR) -> LxR

    M_coo = M.tocoo()
    if M_coo.nnz == 0:
        return pd.DataFrame(columns=["leftNode","rightNode", "pred"])

    lefts = left_nodes.values[M_coo.row]
    rights = right_nodes.values[M_coo.col]
    preds = M_coo.data
    fim = time.time()
    total_time = fim - inicio

    df_out = pd.DataFrame({"leftNode": lefts, "rightNode": rights, "pred": preds})

    df_out = df_out[df_out["leftNode"] != df_out["rightNode"]]
    df_out = df_out.sort_values("pred", ascending=False).reset_index(drop=True)

    df_out = df_out[["leftNode","rightNode", "pred"]]
    return [df_out,total_time]

def getGD(train_graph_path,
           max_dist: int = 5,
           alpha: float = 1.0,
           exclude_existing: bool = True,
           normalize: bool = False):
    """
    Bipartite-aware GD:
      - builds undirected adjacency from training edges (string nodes)
      - BFS from each left node up to max_dist
      - records distances to right-side nodes only and only odd-length distances
      - score = exp(-alpha * (dist-1))  (dist=1 -> score=1.0)
      - can exclude edges that already exist in training (exclude_existing=True)
      - optional min-max normalization across scores (normalize=True)

    Returns DataFrame with columns ["rightNode","leftNode","pred"] sorted by pred desc.
    """

    g = pd.read_csv(train_graph_path, header=None)
    if g.shape[1] < 2:
        raise ValueError("train_graph must have at least two columns (left,right)")
    g = g.rename(columns={0: "leftNode", 1: "rightNode"})
    inicio = time.time()
    g["leftNode"] = g["leftNode"].astype(int)
    g["rightNode"] = g["rightNode"].astype(int)

    left_nodes = list(g["leftNode"].unique())
    right_nodes = list(g["rightNode"].unique())
    left_set = set(left_nodes)
    right_set = set(right_nodes)

    # adjacency (undirected)
    adj = defaultdict(set)
    existing_edges = set()
    for ln, rn in zip(g["leftNode"].values, g["rightNode"].values):
        adj[ln].add(rn)
        adj[rn].add(ln)
        existing_edges.add((ln, rn))

    rows = []
    for ln in left_nodes:
        # BFS from ln
        q = deque()
        q.append((ln, 0))
        dist = {ln: 0}
        while q:
            node, d = q.popleft()
            if d >= max_dist:
                continue
            for nb in adj.get(node, ()):
                if nb not in dist:
                    nd = d + 1
                    dist[nb] = nd
                    q.append((nb, nd))
        # collect right-node distances (only odd distances correspond to left->right paths)
        for rn, d in dist.items():
            if rn in right_set and rn != ln and d >= 1 and (d % 2 == 1):
                if exclude_existing and (ln, rn) in existing_edges:
                    continue
                score = 1/d
                rows.append({"leftNode": ln, "rightNode": rn, "pred": score})

    if not rows:
        return pd.DataFrame(columns=["leftNode","rightNode", "pred"])
    fim = time.time()
    total_time = fim - inicio
    df_out = pd.DataFrame(rows)
    # optional normalization to [0,1]
    if normalize:
        minv = df_out["pred"].min()
        maxv = df_out["pred"].max()
        if maxv > minv:
            df_out["pred"] = (df_out["pred"] - minv) / (maxv - minv)

    df_out = df_out.sort_values("pred", ascending=False).reset_index(drop=True)
    df_out = df_out[["leftNode","rightNode", "pred"]]
    return [df_out,total_time]

def getKatz(train_graph_path,
            max_path_len: int = 5,
            beta: float = 0.05,
            symmetrize: bool = True,
            threshold: float = 1e-6): # <-- Novo parâmetro de segurança
    """
    Compute Katz index scores memory-efficiently.
    """
    # 1. Carregamento e Preparação (Sem alterações)
    g = pd.read_csv(train_graph_path, header=None)
    if g.shape[1] < 2:
        raise ValueError("train_graph must have at least two columns (left,right)")
    g = g.rename(columns={0: "leftNode", 1: "rightNode"})
    g["leftNode"] = g["leftNode"].astype(int)
    g["rightNode"] = g["rightNode"].astype(int)

    left_list = list(g["leftNode"].unique())
    right_list = list(g["rightNode"].unique())

    nodes = pd.Index(pd.concat([g["leftNode"], g["rightNode"]]).unique())
    nodes_arr = nodes.values
    n = len(nodes_arr)
    idx = {node: i for i, node in enumerate(nodes_arr)}

    rows = g["leftNode"].map(idx).values
    cols = g["rightNode"].map(idx).values
    data = np.ones(len(rows), dtype=float)
    inicio = time.time()
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    if symmetrize:
        A = A + A.T
        A.data = np.clip(A.data, 0, 1)

    # 2. Loop do Katz Otimizado com Pruning
    S = sp.csr_matrix((n, n), dtype=float)
    A_pow = A.copy()
    
    for l in range(1, max_path_len + 1):
        # Adiciona à série Katz
        term = A_pow.copy()
        term.data *= (beta ** l)
        S = S + term
        
        # Multiplica para o próximo passo, exceto no último
        if l < max_path_len:
            A_pow = A_pow.dot(A)
            
            # --- SALVA-VIDAS DA MEMÓRIA ---
            # Zera predições quase irrelevantes e limpa a memória estrutural da matriz
            A_pow.data[A_pow.data < threshold] = 0
            A_pow.eliminate_zeros()

    # 3. Fatiamento de Memória (Extrair apenas a parte Bipartida ANTES do Pandas)
    # Descobrimos quais são os índices numéricos das linhas e colunas que queremos
    left_mask = np.isin(nodes_arr, left_list)
    right_mask = np.isin(nodes_arr, right_list)
    
    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]

    # Fatiamos a matriz esparsa. S_bip conterá APENAS pares leftNode -> rightNode
    S_bip = S[left_idx, :][:, right_idx]
    S_coo = S_bip.tocoo()

    if S_coo.nnz == 0:
        return pd.DataFrame(columns=["leftNode", "rightNode", "pred"])
    fim = time.time()
    total_time = fim - inicio
    # 4. Mapeamento de volta para os IDs originais
    lefts = nodes_arr[left_idx[S_coo.row]]
    rights = nodes_arr[right_idx[S_coo.col]]
    preds = S_coo.data

    # 5. Geração do DataFrame Enxuto
    df_out = pd.DataFrame({"leftNode": lefts, "rightNode": rights, "pred": preds})

    # Remoção de self-loops
    df_out = df_out[df_out["leftNode"] != df_out["rightNode"]]

    # Ordenação final
    df_out = df_out.sort_values("pred", ascending=False).reset_index(drop=True)

    return [df_out,total_time]



from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
import numpy as np

def evaluateAllModels(args,
                    ourmethod_dir,
                    train_graph_path,
                    test_graph_path,
                    random_state: int = 0,
                    verbose: bool = True):
    rng = np.random.RandomState(random_state)

    results = {
        "l3": {}, "katz": {}, "common_neighbors": {}, "GD": {},
        "Rf": {}
    }

    # --- test ground-truth ---
    df_test = readFormatExcludedlinks(test_graph_path)
    df_test["leftNode"] = df_test["leftNode"].astype(int)
    df_test["rightNode"] = df_test["rightNode"].astype(int)
    df_test["_exists"] = 1


    def eval_model_heuristic(df_features, feature_col, train_graph_path, df_test, rng=None):
        """
        Avalia métodos heurísticos (L3, Katz, CN) usando seus scores brutos.
        Garante a mesma amostragem de dados e produto cartesiano da função com Random Forest,
        mas avalia apenas como um ranqueamento não-supervisionado.
        """
        try:
            random_seed = int(rng.integers(0, 100000)) if rng else 42
        except:
            random_seed = 42

        # --- 1. DESEMPACOTAMENTO SEGURO ---
        time_val = df_features[1] 
        df_preds = df_features[0]


        # --- 2. CARREGAR GROUND-TRUTH ---
        train_raw = pd.read_csv(train_graph_path, header=None, names=["leftNode", "rightNode"]).astype(int)
        train_raw["is_train"] = 1
        train_raw["label"] = 1 # Arestas originais

        pos_test = df_test[df_test["_exists"] == 1][["leftNode", "rightNode"]].copy()
        pos_test["is_train"] = 0
        pos_test["label"] = 1 # Arestas de teste a serem previstas

        all_positives = pd.concat([train_raw, pos_test])

        # --- 3. PRODUTO CARTESIANO (UNIVERSO DE ARESTAS) ---
        all_left = np.unique(np.concatenate([df_preds['leftNode'], all_positives['leftNode']]))
        all_right = np.unique(np.concatenate([df_preds['rightNode'], all_positives['rightNode']]))
        
        index = pd.MultiIndex.from_product([all_left, all_right], names=['leftNode', 'rightNode'])
        df_merged = pd.DataFrame(index=index).reset_index()

        # --- 4. MERGE E ROTULAÇÃO ---
        # Mapeia quem é positivo (1) e quem é negativo (0)
        df_merged = df_merged.merge(all_positives, on=['leftNode', 'rightNode'], how='left')
        df_merged['label'] = df_merged['label'].fillna(0)
        
        # Mapeia os scores da heurística. O que a heurística não calculou vira 0.0
        df_merged = df_merged.merge(df_preds[['leftNode', 'rightNode', feature_col]], on=['leftNode', 'rightNode'], how='left')
        if feature_col in df_merged.columns:
            df_merged[feature_col] = df_merged[feature_col].fillna(0.0)

        # --- 5. ISOLAR O CONJUNTO DE TESTE ---
        # Heurísticas não são avaliadas nas arestas de treino (pois seu objetivo é prever novos links)
        # Então jogamos o treino fora (is_train == 1) e avaliamos no que sobrou
        df_test_pool = df_merged[df_merged['is_train'] != 1].copy()
        
        df_test_pos = df_test_pool[df_test_pool['label'] == 1].copy()
        df_test_neg_all = df_test_pool[df_test_pool['label'] == 0].copy()

        # --- 6. AVALIAÇÃO NOS CENÁRIOS ---
        results = {}
        cenarios = [("1:1", 1), ("1:10", 10), ("Full", "full")]
        
        for nome, ratio in cenarios:
            if ratio == "full":
                test_sampled = pd.concat([df_test_pos, df_test_neg_all])
            else:
                # Amostra exatamente a mesma quantidade de negativos que a função da RF usaria
                n_neg_test = min(len(df_test_pos) * ratio, len(df_test_neg_all))
                test_neg_sampled = df_test_neg_all.sample(n=n_neg_test, random_state=random_seed)
                test_sampled = pd.concat([df_test_pos, test_neg_sampled])
                
            y_test = test_sampled['label']
            y_score = test_sampled[feature_col] # Usa o SCORE BRUTO (ex: 42612 no CN ou 180.9 no L3)
            
            if len(np.unique(y_test)) < 2:
                print(f"❌ Cenário {nome}: Dados insuficientes.")
                continue
                
            aupr = average_precision_score(y_test, y_score)
            auroc = roc_auc_score(y_test, y_score)
            
            print(f"📊 Heurística {nome.ljust(4)} -> AUPR: {aupr:.5f} | AUROC: {auroc:.5f} (Pares: {len(test_sampled):,})")
            results[nome] = {"aupr": aupr, "auroc": auroc, "time": time_val}
            
        return results

    def eval_model(df_features, feature_cols, train_graph_path, df_test, rng=None):
        """
        Avalia a Random Forest ancorada SEMPRE no ground-truth, 
        evitando perda de dados se o algoritmo omitir predições.
        """
        try:
            random_seed = int(rng.integers(0, 100000)) if rng else 42
        except:
            random_seed = 42

        time_val = df_features[1]
        df_preds = df_features[0] # Renomeado para clareza (são apenas as predições)

        # 1. CARREGAR GROUND-TRUTH (Independente das predições)
        train_raw = pd.read_csv(train_graph_path, header=None, names=["leftNode", "rightNode"]).astype(int)
        train_raw["is_train"] = 1
        train_raw["label"] = 1 # Arestas de treino são positivas

        pos_test = df_test[df_test["_exists"] == 1][["leftNode", "rightNode"]].copy()
        pos_test["is_train"] = 0
        pos_test["label"] = 1 # Arestas de teste são positivas

        all_positives = pd.concat([train_raw, pos_test])

        # 2. PRODUTO CARTESIANO (O UNIVERSO)
        all_left = np.unique(np.concatenate([df_preds['leftNode'], all_positives['leftNode']]))
        all_right = np.unique(np.concatenate([df_preds['rightNode'], all_positives['rightNode']]))
        
        print(f"Gerando Universo: {len(all_left)} nós à esquerda x {len(all_right)} nós à direita...")
        
        index = pd.MultiIndex.from_product([all_left, all_right], names=['leftNode', 'rightNode'])
        df_merged = pd.DataFrame(index=index).reset_index()

        # 3. MERGE DOS LABELS REAIS (O que não é positivo conhecido, é negativo=0)
        df_merged = df_merged.merge(all_positives, on=['leftNode', 'rightNode'], how='left')
        df_merged['label'] = df_merged['label'].fillna(0)
        
        # 4. MERGE DAS PREDIÇÕES (Se o L3 não previu, score é 0.0)
        df_merged = df_merged.merge(df_preds, on=['leftNode', 'rightNode'], how='left')
        if "pred" in df_merged.columns:
            df_merged["pred"] = df_merged["pred"].fillna(0.0)

        # 5. MAPEAMENTO DE GRAUS (Garante que nós não previstos pelo modelo mantenham seus graus corretos)
        left_degrees = train_raw["leftNode"].value_counts().to_dict()
        right_degrees = train_raw["rightNode"].value_counts().to_dict()
        
        # Usamos log1p para evitar erro de log(0) caso algum nó só exista no teste
        df_merged["leftdegree"] = np.log1p(df_merged["leftNode"].map(left_degrees).fillna(0))
        df_merged["rightdegree"] = np.log1p(df_merged["rightNode"].map(right_degrees).fillna(0))

        # Preenchimento de segurança para outras features
        for col in feature_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0.0)

        # 6. DIVISÃO DE TREINO / TESTE E AMOSTRAGEM
        df_train_pos = df_merged[(df_merged['label'] == 1) & (df_merged['is_train'] == 1)].copy()
        df_test_pos = df_merged[(df_merged['label'] == 1) & (df_merged['is_train'] == 0)].copy()
        
        df_negatives_all = df_merged[df_merged['label'] == 0].copy()
        
        num_train_pos = len(df_train_pos)
        if num_train_pos == 0:
            print("❌ ERRO CRÍTICO: Zero arestas de treino encontradas!")
            return {"full": {"aupr": 0, "auroc": 0}, "1:1": {"aupr": 0, "auroc": 0}, "1:10": {"aupr": 0, "auroc": 0}}
            
        df_train_neg = df_negatives_all.sample(n=num_train_pos, random_state=random_seed)
        df_train_neg['is_train'] = 1
        
        df_test_neg_pool = df_negatives_all.drop(df_train_neg.index)
        df_test_neg_pool['is_train'] = 0
        
        # 7. TREINAMENTO (1:1 Balanceado)
        train_balanced = pd.concat([df_train_pos, df_train_neg])
        X_train = train_balanced[feature_cols]
        y_train = train_balanced['label']
        
        print(f"🌲 Treinando RF com {len(train_balanced)} exemplos...")
        rf = LinearDiscriminantAnalysis()
        rf.fit(X_train, y_train.values)
        
        # 8. AVALIAÇÃO
        results = {}
        cenarios = [("1:1", 1), ("1:10", 10), ("Full", "full")]
        
        for nome, ratio in cenarios:
            if ratio == "full":
                test_sampled = pd.concat([df_test_pos, df_test_neg_pool])
            else:
                n_neg_test = min(len(df_test_pos) * ratio, len(df_test_neg_pool))
                test_neg_sampled = df_test_neg_pool.sample(n=n_neg_test, random_state=random_seed)
                test_sampled = pd.concat([df_test_pos, test_neg_sampled])
                
            X_test = test_sampled[feature_cols]
            y_test = test_sampled['label']
            
            if len(np.unique(y_test)) < 2:
                continue
                
            y_score = rf.predict_proba(X_test)[:, 1]
            
            aupr = average_precision_score(y_test, y_score)
            auroc = roc_auc_score(y_test, y_score)
            
            print(f"✅ {nome.ljust(5)} -> AUPR: {aupr:.5f} | AUROC: {auroc:.5f} (Pares: {len(test_sampled):,})")
            results[nome] = {"aupr": aupr, "auroc": auroc, "time": time_val}
            
        return results
       
        
    # Retorna exatamente a estrutura esperada pelo script de Experimentos

    try:
        df, time_val = getKatz(train_graph_path=train_graph_path)
        results["katz"] = eval_model_heuristic([df, time_val], "pred", train_graph_path, df_test, rng)
    except Exception as e:
        results["katz"] = {"full": {"aupr": None, "auroc": None}, "1:1": {"aupr": None, "auroc": None}, "1:10": {"aupr": None, "auroc": None}}
        if verbose: 
            print("Error evaluating Katz:", e)
    try:
        df, time_val = getL3(args,train_graph_path=train_graph_path)
        results["L3"] = eval_model_heuristic([df, time_val],"pred", train_graph_path, df_test, rng)
    except Exception as e:
        results["L3"] = {"full": {"aupr": None, "auroc": None}, "1:1": {"aupr": None, "auroc": None}, "1:10": {"aupr": None, "auroc": None}}
        if verbose: 
            print("Error evaluating L3:", e)
   

    try:
        df,time = getCommonNeighbors(train_graph_path=train_graph_path)
        results["common_neighbors"] = eval_model_heuristic(
                                        [df,time],
                                        "pred", train_graph_path, df_test, rng)
    except Exception as e:
        results["common_neighbors"] = {"full": {"aupr": None, "auroc": None}, "1:1": {"aupr": None, "auroc": None}, "1:10": {"aupr": None, "auroc": None}}
        if verbose: 
            print("Error evaluating CommonNeighbors:", e)


    try:
        df, time_val = getGD(train_graph_path=train_graph_path)
        results["GD"] = eval_model_heuristic([df, time_val], "pred", train_graph_path, df_test, rng)
    except Exception as e:

        results["GD"] = {"full": {"aupr": None, "auroc": None}, "1:1": {"aupr": None, "auroc": None}, "1:10": {"aupr": None, "auroc": None}}
        if verbose: 
            print("Error evaluating GD:", e)

    try:
        if verbose: print("A avaliar o Espaço de Features (Normalização ancorada no Treino + Penalização)...")
        
        train_set, test_set = build_sets(train_graph_path, test_graph_path)
        train_df = pd.DataFrame(train_set, columns=['col',"row1"])
        train_df['is_train'] = True
        train_df['row1'] = train_df['row1'].astype(int)
        train_df['col'] = train_df['col'].astype(int)
        
        test_df = pd.DataFrame(test_set, columns=['col',"row1"])
        test_df['is_test'] = True
        test_df['row1'] = test_df['row1'].astype(int)
        test_df['col'] = test_df['col'].astype(int)
        
        if verbose: print("  -> A mapear os graus de treino...")
        row_degrees = train_df['row1'].value_counts().to_dict()
        col_degrees = train_df['col'].value_counts().to_dict()
        
        dtype_features = np.dtype([
            ('col', np.int32), 
            ('row1', np.int32), 
            ('kurt_skew', np.float32), 
            ('entropy', np.float32)
        ])
        
        df_calc= pd.DataFrame(np.fromfile(os.path.join(ourmethod_dir,"1_final_kurtosis_output.bin"), dtype=dtype_features,))

        

        if verbose: print("  -> A normalizar dados e calcular distâncias...")
        

        df_calc["col"] = df_calc["col"].astype(np.int64)
        df_calc["row1"] = df_calc["row1"].astype(np.int64)

        train_df["col"] = train_df["col"].astype(np.int64)
        train_df["row1"] = train_df["row1"].astype(np.int64)

        row_degrees = train_df['row1'].value_counts().to_dict()
        col_degrees = train_df['col'].value_counts().to_dict()

        df_calc['deg_row'] = df_calc['row1'].map(row_degrees)
        df_calc['deg_col'] = df_calc['col'].map(col_degrees)



        df_calc['x_min'] = df_calc.groupby('col', as_index=False)['kurt_skew'].transform('min')
        df_calc['x_max'] = df_calc.groupby('col', as_index=False)['kurt_skew'].transform('max')
        df_calc['y_min'] = df_calc.groupby('col', as_index=False)['entropy'].transform('min')
        df_calc['y_max'] = df_calc.groupby('col', as_index=False)['entropy'].transform('max')

        df_calc['x_range'] = (df_calc['x_max'] - df_calc['x_min']).replace(0, 1)
        df_calc['y_range'] = (df_calc['y_max'] - df_calc['y_min']).replace(0, 1)
        
        df_calc['x_norm'] = (df_calc['kurt_skew'] - df_calc['x_min']) / df_calc['x_range']
        df_calc['y_norm'] = (df_calc['entropy'] - df_calc['y_min']) / df_calc['y_range']
        

        print("x_norm finite:", np.isfinite(df_calc['x_norm']).mean())
        print("y_norm finite:", np.isfinite(df_calc['y_norm']).mean())
        
        # 2. Distância Base para (0,1)
        distancia_base = np.sqrt((df_calc['x_norm'] - 0.0)**2 + (df_calc['y_norm'] - 1.0)**2)
        
        # 3. Penalização Proporcional aos Graus
        df_calc['deg_row'] = df_calc['row1'].map(row_degrees).fillna(0)
        df_calc['deg_col'] = df_calc['col'].map(col_degrees).fillna(0)
        
        
        df_calc['dist_final'] = 1- distancia_base   
        df_calc["priority"] = df_calc["dist_final"]
        
        df_calc = df_calc.rename(columns={"col":"leftNode",
                                "row1":"rightNode",
                                "priority":"pred"})





    
        # 3. Aplicar a calibração por nó

        results["RF"] = eval_model([df_calc,1],["rightdegree","leftdegree","entropy","kurt_skew"], train_graph_path, df_test, rng)

        print(results)
    except Exception as e:
        results["RF"] = {"full": {"aupr": None, "auroc": None}, "1:1": {"aupr": None, "auroc": None}, "1:10": {"aupr": None, "auroc": None}}
        if verbose: 
            print("Error evaluating RF:", e)
    return results


def readFormatExcludedlinks(adress:str):

    dfExcluded = pd.read_csv(adress,header=None)
    dfExcluded.rename(columns={0:"leftNode",
                               1:"rightNode"},inplace=True)
    dfExcluded["leftNode"] = dfExcluded["leftNode"].astype("int64")
    dfExcluded["rightNode"] = dfExcluded["rightNode"].astype("int64")

    
    return dfExcluded

def run_ablation_study(df_train, df_test, sn, mode, random_seed=42):
    """
    Carrega os dados, filtra as features de acordo com o sn e mode,
    treina a rede neural e avalia nos cenários de desbalanceamento.
    """
    print(f"\n--- Iniciando Avaliação | SN: {sn} | MODE: {mode} ---")
    

    # 2. Definição dos blocos de features
    f_deg = ['leftDegree', 'rightDegree']
    f_kse = ['entropy', 'kurt_skew']
    f_hist = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9']

    # 3. Mapeamento dos cenários de ablação com base nos parâmetros
    avaliacoes = []
    
    if sn == 1 and mode == 0:
        avaliacoes.append(("1_Somente_Degrees", f_deg))
        avaliacoes.append(("2_Somente_Kurt_Skew_Entropy", f_kse))
        avaliacoes.append(("4_Degree_Ent_Kurt_Skew_SN1", f_deg + f_kse))
    elif sn == 1 and mode == 2:
        avaliacoes.append(("3_Somente_Hist", f_hist))
        avaliacoes.append(("5_Degree_Hist_SN1", f_deg + f_hist))
    elif sn == 2 and mode == 0:
        avaliacoes.append(("6_Degree_Ent_Kurt_Skew_SN2", f_deg + f_kse))
    elif sn == 2 and mode == 2:
        avaliacoes.append(("7_Degree_Hist_SN2", f_deg + f_hist))

    # Preparar a amostragem de teste para os cenários (1:1, 1:10, Full)
    df_test_pos = df_test[df_test['label'] == 1]
    df_test_neg_pool = df_test[df_test['label'] == 0]

    resultados_ablation = []

    # 4. Loop de Treino e Avaliação
    for nome_teste, feature_cols in avaliacoes:
        colunas_presentes = [c for c in feature_cols if c in df_train.columns]
        if len(colunas_presentes) < len(feature_cols):
            print(f"⚠️ Aviso: Faltam colunas para {nome_teste}.")
            continue

        print(f" Treinando modelo para ablação: {nome_teste}")
        X_train = df_train[colunas_presentes].values
        y_train = df_train['label'].values

        # Inicializa e treina o nosso wrapper da MLP com Scaler embutido
        rf = SklearnMLPWrapper(epochs=100, batch_size=512)
        rf.fit(X_train, y_train)

        # 5. Loop dos 3 Cenários (1:1, 1:10, Full)
        cenarios_proporcao = [("1:1", 1), ("1:10", 10), ("Full", "full")]
        
        for nome_cenario, ratio in cenarios_proporcao:
            if ratio == "full":
                test_sampled = df_test
            else:
                n_neg_test = min(len(df_test_pos) * ratio, len(df_test_neg_pool))
                test_neg_sampled = df_test_neg_pool.sample(n=n_neg_test, random_state=random_seed)
                test_sampled = pd.concat([df_test_pos, test_neg_sampled])
                
            X_test = test_sampled[colunas_presentes].values
            y_test = test_sampled['label'].values

            if len(np.unique(y_test)) < 2:
                continue
                
            y_score = rf.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, y_score)
            aupr = average_precision_score(y_test, y_score)

            resultados_ablation.append({
                "Model": nome_teste,
                "Scenario": nome_cenario,
                "AUPR": aupr,
                "AUROC": auroc
            })
            
            print(f"  -> Cenário {nome_cenario} | AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

    # Limpeza de memória
    del df_train, df_test, df_test_pos, df_test_neg_pool
    gc.collect()

    return resultados_ablation

def evaluateOurModels(args,
                      ourmethod_dir,
                      train_graph_path,
                      test_graph_path,
                      random_state: int = 0,
                      mode: int = 2):
    """
    Avalia o método (OurMethod) treinado com Random Forest.
    Suporta Mode 0 (kurt_skew + entropy) ou Mode 2 (10 Bins/Decis).
    """
    try:
        rng = np.random.RandomState(random_state)
        random_seed = int(rng.integers(0, 100000))
    except:
        random_seed = 42

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Dispositivo selecionado: {device.upper()}")
    print(f"Iniciando avaliação OurMethod (Mode {mode}) em {ourmethod_dir}...")

    # --- 1. LER OS DADOS BINÁRIOS (STRUCT DO C++) ---
    if mode == 2:
        # Modo 2: Mapeia perfeitamente a struct OutputEdgeHist do C++
        dtype_our = np.dtype([
            ('col', np.int32),
            ('row1', np.int32),
            ('bin0', np.float32), ('bin1', np.float32), ('bin2', np.float32), 
            ('bin3', np.float32), ('bin4', np.float32), ('bin5', np.float32),
            ('bin6', np.float32), ('bin7', np.float32), ('bin8', np.float32), 
            ('bin9', np.float32)
        ])
        feature_cols = [f'bin{i}' for i in range(10)] + ['leftDegree', 'rightDegree']
        file_pattern = '*_final_histograms_output.bin'
    elif mode == 1:
        dtype_our = np.dtype([
            ('col', np.int32),
            ('row1', np.int32),
            ('kurt_skew', np.float32), 
            ('entropy', np.float32)
        ])
        feature_cols = ['leftDegree', 'rightDegree']
        file_pattern = '*_final_kurtosis_output.bin'
       
    else:
        # Modo 0: Estrutura antiga
        dtype_our = np.dtype([
            ('col', np.int32),
            ('row1', np.int32),
            ('kurt_skew', np.float32), 
            ('entropy', np.float32)
        ])
        feature_cols = ['kurt_skew', 'entropy']
        file_pattern = '*_final_kurtosis_output.bin'


    df_list = []
    # Usando np.fromfile em vez de pd.read_csv para leitura binária segura
    for file_path in glob.glob(os.path.join(ourmethod_dir, file_pattern)):
        data = np.fromfile(file_path, dtype=dtype_our)
        df_list.append(pd.DataFrame(data))

    if not df_list:
        print(f"❌ Nenhum ficheiro encontrado em {ourmethod_dir} com o padrão {file_pattern}")
        return {"full": {"aupr": 0, "auroc": 0}, "1:1": {"aupr": 0, "auroc": 0}, "1:10": {"aupr": 0, "auroc": 0}}

    df_preds = pd.concat(df_list, ignore_index=True)
    df_preds = df_preds.rename(columns={"col": "leftNode", "row1": "rightNode"})

    # --- 2. CARREGAR GROUND-TRUTH E MONTAR UNIVERSO ---
    train_raw = pd.read_csv(train_graph_path, header=None, names=["leftNode", "rightNode"]).astype(int)
    train_raw["is_train"] = 1
    train_raw["label"] = 1 

    df_test = readFormatExcludedlinks(test_graph_path)
    df_test["leftNode"] = df_test["leftNode"].astype(int)
    df_test["rightNode"] = df_test["rightNode"].astype(int)
    df_test["_exists"] = 1
    
    pos_test = df_test[df_test["_exists"] == 1][["leftNode", "rightNode"]].copy()
    pos_test["is_train"] = 0
    pos_test["label"] = 1

    all_positives = pd.concat([train_raw, pos_test])

    # Cria o produto cartesiano apenas com os nós que apareceram
    all_left = np.unique(np.concatenate([df_preds['leftNode'], all_positives['leftNode']]))
    all_right = np.unique(np.concatenate([df_preds['rightNode'], all_positives['rightNode']]))
    
    print(f"Gerando Universo: {len(all_left)} nós à esquerda x {len(all_right)} nós à direita...")
    index = pd.MultiIndex.from_product([all_left, all_right], names=['leftNode', 'rightNode'])
    df_merged = pd.DataFrame(index=index).reset_index()

    # --- 3. MERGE DE INFORMAÇÕES E GRAUS ---
    df_merged = df_merged.merge(all_positives, on=['leftNode', 'rightNode'], how='left')
    df_merged['label'] = df_merged['label'].fillna(0)
    
    df_merged = df_merged.merge(df_preds, on=['leftNode', 'rightNode'], how='left')
    
    # Ancoragem topológica: O modelo saberá o grau do nó base para contextualizar as features
    left_degrees = train_raw["leftNode"].value_counts().to_dict()
    right_degrees = train_raw["rightNode"].value_counts().to_dict()
    df_merged["leftDegree"] = np.log1p(df_merged["leftNode"].map(left_degrees).fillna(0))
    df_merged["righDdegree"] = np.log1p(df_merged["rightNode"].map(right_degrees).fillna(0))

    # Preenche com zeros pares de nós que não alcançaram qualquer permutação
    for col in feature_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0.0)

    # --- 4. DIVISÃO TREINO/TESTE (Negative Sampling) ---
    df_train_pos = df_merged[(df_merged['label'] == 1) & (df_merged['is_train'] == 1)].copy()
    df_test_pos = df_merged[(df_merged['label'] == 1) & (df_merged['is_train'] == 0)].copy()
    df_negatives_all = df_merged[df_merged['label'] == 0].copy()
    
    num_train_pos = len(df_train_pos)
    if num_train_pos == 0:
        print("❌ ERRO CRÍTICO: Zero arestas de treino encontradas!")
        return {"full": {"aupr": 0, "auroc": 0}, "1:1": {"aupr": 0, "auroc": 0}, "1:10": {"aupr": 0, "auroc": 0}}

    # Amostra negativos para equilibrar o treino
    df_train_neg = df_negatives_all.sample(n=num_train_pos, random_state=random_seed)
    df_train_neg['is_train'] = 1
    
    df_test_neg_pool = df_negatives_all.drop(df_train_neg.index)
    df_test_neg_pool['is_train'] = 0

    # --- 5. TREINAR O CLASSIFICADOR (Random Forest) ---
    train_balanced = pd.concat([df_train_pos, df_train_neg])
    X_train = train_balanced[feature_cols].values  #
    y_train = train_balanced['label'].values  #
    
    print(f"🌲 Treinando Random Forest com {len(train_balanced)} exemplos ({len(feature_cols)} features)...")

    # Lendo tempo de execução (se existir um ficheiro time_.csv)
    time_val = 0.0
    time_files = glob.glob(os.path.join(ourmethod_dir, 'time_*.csv'))
    if time_files:
        with open(time_files[0], 'r') as f:
            try:
                time_val = float(f.read().strip())
            except:
                pass

    # --- 6. AVALIAR O MODELO NOS 3 CENÁRIOS ---
    results = {}
    cenarios = [("1:1", 1), ("1:10", 10), ("Full", "full")]
    
    for nome, ratio in cenarios:
        if ratio == "full":
            test_sampled = pd.concat([df_test_pos, df_test_neg_pool])
        else:
            n_neg_test = min(len(df_test_pos) * ratio, len(df_test_neg_pool))
            test_neg_sampled = df_test_neg_pool.sample(n=n_neg_test, random_state=random_seed)
            test_sampled = pd.concat([df_test_pos, test_neg_sampled])
            
        X_test = test_sampled[feature_cols].values  #
        y_test = test_sampled['label'].values  #
        rf = SklearnMLPWrapper(epochs=100, batch_size=512)
        rf.fit(X_train, y_train)
        if len(np.unique(y_test)) < 2:
            print(f"❌ Cenário {nome}: Dados insuficientes.")
            continue
            
        # Pega a probabilidade de pertencer à Classe 1 (Link)
        y_score = rf.predict_proba(X_test)[:, 1]
        
        aupr = average_precision_score(y_test, y_score)
        auroc = roc_auc_score(y_test, y_score)
        
        print(f"📊 OurMethod {nome.ljust(4)} -> AUPR: {aupr:.5f} | AUROC: {auroc:.5f} (Pares: {len(test_sampled):,})")
        results[nome] = {"aupr": aupr, "auroc": auroc, "time": time_val}

    return results