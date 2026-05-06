import torch
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
import time
import os
from create_graph import create_graph
import subprocess
import xgboost as xgb
import numpy as np
import pandas as pd
import os, subprocess, gc
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from args import *
import xgboost as xgb
import numpy as np
import pandas as pd
import os, subprocess, gc, time
import numpy as np
import scipy.sparse as sp
import os
import time
from collections import deque, defaultdict
import pandas as pd
import numpy as np
import os

def readFormatExcludedlinks(adress:str):

    dfExcluded = pd.read_csv(adress,header=None)
    dfExcluded.rename(columns={0:"leftNode",
                               1:"rightNode"},inplace=True)
    dfExcluded["leftNode"] = dfExcluded["leftNode"].astype("int64")
    dfExcluded["rightNode"] = dfExcluded["rightNode"].astype("int64")

    
    return dfExcluded



def evaluateBaselines(args):
    
    base_dir = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0], args.SampleTecnic)
    trainFile = os.path.join(base_dir, f"trainGraph{args.CurrentSeed}.csv")
    testFile = os.path.join(base_dir, f"testGraph{args.CurrentSeed}.csv")
    targetFile = os.path.join(base_dir, f"targetGraph{args.CurrentSeed}_0.csv")


    df_targets = pd.read_csv(targetFile, header=None, names=["leftNode", "rightNode"])
    
    df_train = pd.read_csv(trainFile, header=None, names=["leftNode", "rightNode"])
    df_train['is_train'] = True
    
    df_test = pd.read_csv(testFile, header=None, names=["leftNode", "rightNode"])
    df_test['label'] = 1
    
    df_eval_base = pd.merge(df_targets, df_train, on=["leftNode", "rightNode"], how="left")
    df_eval_base = df_eval_base[df_eval_base['is_train'].isna()].drop(columns=['is_train'])
    
    df_eval_base = pd.merge(df_eval_base, df_test, on=["leftNode", "rightNode"], how="left")
    df_eval_base['label'] = df_eval_base['label'].fillna(0).astype(np.int8)
    
    pos_idx_base = np.where(df_eval_base["label"] == 1)[0]
    neg_idx_base = np.where(df_eval_base["label"] == 0)[0]

    print(f"Alvos de Avaliação Prontos: {len(pos_idx_base)} Positivos | {len(neg_idx_base)} Negativos")


    baselines = {
        "Node2Vec": lambda: getNode2Vec(trainFile, targetFile),
        "GraphSAGE": lambda: getGraphSAGE(trainFile, targetFile),
        "Katz": lambda: getKatz(trainFile, max_path_len=3, beta=0.005),
        "L3": lambda: getL3(args, trainFile),
        "GD": lambda: getGD(trainFile, max_dist=5)
    }

    resultados_baselines = []

    # =========================================================================
    # 3. AVALIAÇÃO FORÇADA NOS ALVOS
    # =========================================================================
    for name, func in baselines.items():
        try:
            df_pred, exec_time = func()
            
            if df_pred.empty:
                print(f" {name} não gerou predições.")
                continue

            df_pred["leftNode"] = df_pred["leftNode"].astype(np.int32)
            df_pred["rightNode"] = df_pred["rightNode"].astype(np.int32)

            df_model = pd.merge(df_eval_base, df_pred[['leftNode', 'rightNode', 'pred']], 
                                on=['leftNode', 'rightNode'], how='left')
            
            df_model['pred'] = df_model['pred'].fillna(0).astype(np.float32)

            y_true = df_model["label"].values
            y_scores = df_model["pred"].values

            rng = np.random.default_rng(args.CurrentSeed)

            for ratio_name, ratio in [("1:1", 1), ("1:10", 10), ("1:100", 100)]:
                n_neg = min(len(pos_idx_base) * ratio, len(neg_idx_base))
                neg_sample = rng.choice(neg_idx_base, n_neg, replace=False)
                
                idx = np.concatenate([pos_idx_base, neg_sample])

                yt = y_true[idx]
                ys = y_scores[idx]
                
                auroc = roc_auc_score(yt, ys)
                aupr = average_precision_score(yt, ys)

                resultados_baselines.append({
                    "Dataset": args.dataset,
                    "Seed": args.CurrentSeed,
                    "Sampling": args.SampleTecnic,
                    "Features": name, 
                    "Ratio": ratio_name,
                    "AUROC": round(auroc, 4),
                    "AUPR": round(aupr, 4),
                    "Time": round(exec_time, 2),
                    "N_Samples": len(idx)
                })
                
                print(f"  [{name}] {ratio_name} -> AUROC: {auroc:.4f} | AUPR: {aupr:.4f} (N={len(idx)})")

            del df_pred, df_model
            gc.collect()

        except Exception as e:
            print(f"❌ Erro em {name}: {e}")
            

    if resultados_baselines:
        out_dir = os.path.join(args.exec_path, "data", "exp")
        os.makedirs(out_dir, exist_ok=True)
        arquivo_resultados = os.path.join(out_dir, "ResultadosHeuristicas.csv")
        
        df_resultados = pd.DataFrame(resultados_baselines)
        
        arquivo_existe = os.path.isfile(arquivo_resultados)
        df_resultados.to_csv(arquivo_resultados, mode='a', index=False, header=not arquivo_existe)
        
        print(f"\n Resultados das Heurísticas (Seed {args.CurrentSeed}) salvos em: {arquivo_resultados}")
            
    return resultados_baselines



import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import os

def getNode2Vec(train_graph_path, target_graph_path, dimensions=64, epochs=150):
    inicio = time.time()
    
    device = torch.device('cpu')
    num_cores = os.cpu_count() or 1
    torch.set_num_threads(num_cores)
    
    g_df = pd.read_csv(train_graph_path, header=None, names=["leftNode", "rightNode"])
    
    u_nodes = torch.tensor(g_df["leftNode"].values, dtype=torch.long, device=device)
    v_nodes = torch.tensor(g_df["rightNode"].values, dtype=torch.long, device=device)
    
    num_nodes = int(max(u_nodes.max().item(), v_nodes.max().item())) + 1
    
    emb = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=dimensions).to(device)
    torch.nn.init.xavier_uniform_(emb.weight)
    
    optimizer = torch.optim.Adam(emb.parameters(), lr=0.05)
    
    emb.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        emb_u = emb(u_nodes)
        emb_v = emb(v_nodes)
        pos_score = (emb_u * emb_v).sum(dim=1)
        pos_loss = -F.logsigmoid(pos_score).mean() 
        
        neg_v_nodes = torch.randint(0, num_nodes, (len(u_nodes),), device=device)
        emb_neg_v = emb(neg_v_nodes)
        neg_score = (emb_u * emb_neg_v).sum(dim=1)
        neg_loss = -F.logsigmoid(-neg_score).mean()
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
    emb.eval()
    with torch.no_grad():
        z = emb.weight.cpu().detach()
        
    targets_df = pd.read_csv(target_graph_path, header=None, names=["leftNode", "rightNode"])
    
    u_arr = targets_df["leftNode"].values
    v_arr = targets_df["rightNode"].values
    
    valid_mask = (u_arr < num_nodes) & (v_arr < num_nodes)
    
    valid_u = u_arr[valid_mask]
    valid_v = v_arr[valid_mask]
    
    if len(valid_u) > 0:
        scores = (z[valid_u] * z[valid_v]).sum(dim=1).numpy()
    else:
        scores = np.array([])
        
    preds = np.zeros(len(targets_df), dtype=np.float32)
    preds[valid_mask] = scores
    
    targets_df["pred"] = preds
    targets_df.sort_values(by="pred", ascending=False, inplace=True)
    targets_df.reset_index(drop=True, inplace=True)
    
    fim = time.time()
    return [targets_df, fim - inicio]


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def getGraphSAGE(train_graph_path, target_graph_path, epochs=50):
    inicio = time.time()
    
    device = torch.device('cpu')
    num_cores = os.cpu_count() or 1
    torch.set_num_threads(num_cores) 
    
    g_df = pd.read_csv(train_graph_path, header=None, names=["leftNode", "rightNode"])
    
    u_nodes = g_df["leftNode"].tolist()
    v_nodes = g_df["rightNode"].tolist()
    
    edge_index = torch.tensor([u_nodes, v_nodes], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    num_nodes = int(torch.max(edge_index)) + 1
    
    x = torch.eye(num_nodes, dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    
    model = GraphSAGEModel(in_channels=num_nodes, hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(x, edge_index)
        
        pos_out = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
        
    targets_df = pd.read_csv(target_graph_path, header=None, names=["leftNode", "rightNode"])
    
    u_arr = targets_df["leftNode"].values
    v_arr = targets_df["rightNode"].values
    
    valid_mask = (u_arr < num_nodes) & (v_arr < num_nodes)
    
    valid_u = u_arr[valid_mask]
    valid_v = v_arr[valid_mask]
    
    if len(valid_u) > 0:
        z_u = z[valid_u]
        z_v = z[valid_v]
        scores = (z_u * z_v).sum(dim=1).cpu().numpy()
    else:
        scores = np.array([])
        
    preds = np.zeros(len(targets_df), dtype=np.float32)
    preds[valid_mask] = scores
    
    targets_df["pred"] = preds
    
    targets_df.sort_values(by="pred", ascending=False, inplace=True)
    targets_df.reset_index(drop=True, inplace=True)
    
    fim = time.time()
    return [targets_df, fim - inicio]

def runMethod(args):

    rng = np.random.default_rng(args.CurrentSeed)
    np.random.seed(args.CurrentSeed)

    create_graph(args)
    codeFile = os.path.join(args.exec_path,"code", "OurMethodExp.cpp")
    org = os.path.join(args.exec_path,"code", "prob_calc.cpp")
    exe = os.path.join(args.exec_path,"code", "build", "OurMethod")  
    os.makedirs(os.path.dirname(exe), exist_ok=True)

    base_path = os.path.join(
        args.exec_path, "data", "exp",
        args.dataset.split("/")[0],
        args.SampleTecnic,
        str(args.CurrentSeed),
        str(args.Filter)
    )

    os.makedirs(base_path, exist_ok=True)
    resultados_experimento = []
    trainFile = os.path.join(args.exec_path, "data", "exp",
                             args.dataset.split("/")[0],
                             args.SampleTecnic,
                             f"trainGraph{args.CurrentSeed}.csv")

    testFile = os.path.join(args.exec_path, "data", "exp",
                            args.dataset.split("/")[0],
                            args.SampleTecnic,
                            f"testGraph{args.CurrentSeed}.csv")

    targetFile = os.path.join(args.exec_path, "data", "exp",
                              args.dataset.split("/")[0],
                              args.SampleTecnic,
                              f"targetGraph{args.CurrentSeed}_{args.Filter}.csv")
    


    train_raw = pd.read_csv(trainFile, header=None, names=["leftNode", "rightNode"]).astype(np.int32)
    df_test = readFormatExcludedlinks(testFile)
    pos_test = df_test[["leftNode", "rightNode"]].astype(np.int32)

    df_pos = pd.concat([train_raw, pos_test]).drop_duplicates()

    all_users = np.array(df_pos["leftNode"].values, dtype=np.int32, copy=True)
    all_items = np.array(df_pos["rightNode"].values, dtype=np.int32, copy=True)


    print(f"Gerando target (stream Zero-Memory para o dataset {args.dataset})...")
    
    print(f"Gerando target para {args.dataset} (Subgrafo Amostrado)...")
    
    ratio_teste = 100 
    ratio_treino = 10
    # Garante negativos suficientes para o teste 1:100 E para o treino 1:10
    target_neg = int(len(pos_test) * ratio_teste) + int(len(train_raw) * ratio_treino)
    print(len(pos_test))
    print(target_neg)
    df_pos_reais = df_pos[['leftNode', 'rightNode']].copy()
    df_pos_reais['real'] = 1

    unique_users = np.array(all_users)
    unique_items = np.array(all_items)

    with open(targetFile, "w", buffering=1000000) as f:
        df_pos[['leftNode', 'rightNode']].astype(np.int32).to_csv(
            f, index=False, header=False, lineterminator='\n'
        )

        generated = 0
        batch_size = 1_000_000 
        
        while generated < target_neg:
            idx_u = rng.integers(0, len(unique_users), size=batch_size)
            idx_i = rng.integers(0, len(unique_items), size=batch_size)
            
            df_batch = pd.DataFrame({
                'leftNode': unique_users[idx_u],
                'rightNode': unique_items[idx_i]
            })
            
            df_batch = pd.merge(df_batch, df_pos_reais, on=['leftNode', 'rightNode'], how='left')
            df_batch = df_batch[df_batch['real'].isna()][['leftNode', 'rightNode']].drop_duplicates()
            
            needed = target_neg - generated
            if len(df_batch) > needed:
                df_batch = df_batch.iloc[:needed]
            
            if not df_batch.empty:
                df_batch.astype(np.int32).to_csv(f, index=False, header=False, lineterminator='\n')
                generated += len(df_batch)
            
            del df_batch
        df_pos_reais = None
        unique_users = None
        unique_items = None

        compile_cmd = ["g++", "-O3","-pthread","-I", f"{args.exec_path}/eigen-3.4.0", org, codeFile, "-o", exe]
        subprocess.run(compile_cmd, check=True)
        configs = [(1,0), (1,2), (2,0), (2,2)]

        for sn, mode in configs:
            print(f"Rodando C++ SN={sn} MODE={mode}")
            subprocess.run([
                exe, trainFile, base_path,
                str(sn), str(args.CurrentSeed),
                str(mode), targetFile
            ], check=True)

        dtype_hist = np.dtype([
            ('leftNode', np.int32), ('rightNode', np.int32),
            *[(f'bin{i}', np.float32) for i in range(10)]
        ])

        dtype_kse = np.dtype([
            ('leftNode', np.int32), ('rightNode', np.int32),
            ('kurt', np.float32), ('skew', np.float32), ('entropy', np.float32)
        ])

        print("\nAlinhando características...")
        df_features = None

        print("\nAlinhando características...")
    
        df_base = pd.read_csv(targetFile, header=None, names=["leftNode", "rightNode"])

        for idx, sn in enumerate([1, 2]):
            hist_file = os.path.join(base_path, f"{sn}_final_histograms_output.bin")
            kse_file  = os.path.join(base_path, f"{sn}_final_kurtosis_output.bin")

            mem_hist = np.memmap(hist_file, dtype=dtype_hist, mode='r')
            mem_kse  = np.memmap(kse_file, dtype=dtype_kse, mode='r')

            df_h = pd.DataFrame(np.array(mem_hist))
            df_k = pd.DataFrame(np.array(mem_kse))

            del mem_hist, mem_kse
            gc.collect()

            df_h.columns = ['leftNode', 'rightNode'] + [f'bin{i}_{idx}' for i in range(10)]
            df_k.columns = ['leftNode', 'rightNode', f'kurt_{idx}', f'skew_{idx}', f'entropy_{idx}']

            df_config = pd.merge(df_h, df_k, on=['leftNode', 'rightNode'], how='inner')

            df_base = pd.merge(df_base, df_config, on=['leftNode', 'rightNode'], how='left')
                
            del df_h, df_k, df_config
            gc.collect()

        df_features = df_base.fillna(0).astype(np.float32)
        
        df_features['leftNode'] = df_features['leftNode'].astype(np.int32)
        df_features['rightNode'] = df_features['rightNode'].astype(np.int32)

        left_deg = np.bincount(train_raw["leftNode"])
        right_deg = np.bincount(train_raw["rightNode"])
        max_l = len(left_deg) - 1
        max_r = len(right_deg) - 1

        left = df_features['leftNode'].values
        right = df_features['rightNode'].values

        all_features = []
        for idx in range(2): 
            cols_bins = [f'bin{i}_{idx}' for i in range(10)]
            cols_kse = [f'kurt_{idx}', f'skew_{idx}', f'entropy_{idx}']
            
            all_features.append(df_features[cols_bins].values.astype(np.float32))
            all_features.append(df_features[cols_kse].values.astype(np.float32))

        X = np.hstack(all_features)
        
        del df_features, all_features
        gc.collect()

        ldeg = np.log1p(left_deg[np.clip(left, 0, max_l)]).astype(np.float32)
        rdeg = np.log1p(right_deg[np.clip(right, 0, max_r)]).astype(np.float32)

        X = np.hstack([X, ldeg[:, None], rdeg[:, None]], dtype=np.float32)
        
        del left_deg, right_deg, ldeg, rdeg
        gc.collect()

        df_train_keys = train_raw[['leftNode', 'rightNode']].copy()
        df_train_keys['split'] = 'train_pos'

        df_test_keys = pos_test[['leftNode', 'rightNode']].copy()
        df_test_keys['split'] = 'test_pos'

        del train_raw, pos_test, df_test
        gc.collect()

        df_splits = pd.concat([df_train_keys, df_test_keys]).drop_duplicates()

        df_keys_only = pd.DataFrame({'leftNode': left, 'rightNode': right})
        
        del left, right
        gc.collect()

        df_labels = pd.merge(df_keys_only, df_splits, on=['leftNode', 'rightNode'], how='left')
        df_labels['split'] = df_labels['split'].fillna('negative')

        split_col = df_labels['split'].values

        df_labels = pd.merge(df_keys_only, df_splits, on=['leftNode', 'rightNode'], how='left')
        df_labels['split'] = df_labels['split'].fillna('negative')

        split_col = df_labels['split'].values

        is_train_pos = (split_col == 'train_pos')
        is_test_pos = (split_col == 'test_pos')
        is_negative = (split_col == 'negative')

        neg_indices = np.where(is_negative)[0]
        n_train_pos = np.sum(is_train_pos) 
    
        # Amostra 10 vezes mais negativos do que positivos para o treino
        n_train_neg = int(n_train_pos * ratio_treino) 
        train_neg_idx = rng.choice(neg_indices, size=n_train_neg, replace=False)
    
        test_neg_idx = np.array(list(set(neg_indices) - set(train_neg_idx)))
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[is_train_pos] = True
        train_mask[train_neg_idx] = True

        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[is_test_pos] = True
        test_mask[test_neg_idx] = True

        y = np.zeros(len(X), dtype=np.int8)
        y[is_train_pos | is_test_pos] = 1

        X_train = X[train_mask]
        y_train = y[train_mask]
        
        X_test = X[test_mask]
        y_test = y[test_mask]

        del df_keys_only, df_splits, df_labels, X, train_mask, test_mask, split_col
        del df_train_keys, df_test_keys, is_train_pos, is_test_pos, is_negative, neg_indices
        gc.collect()


        feature_map = {}
        offset = 0

        for i in range(2):
            feature_map[f"bins_{i}"] = list(range(offset, offset+10))
            feature_map[f"kse_{i}"] = list(range(offset+10, offset+13))
            offset += 13

        feature_map["deg"] = [offset, offset+1]
        print(feature_map)
        avaliacoes = [
        ("Only_Degree", feature_map["deg"]),
        ("Hist_SN1_Degree", feature_map["bins_0"] + feature_map["deg"]),
        ("Hist_SN2_Degree", feature_map["bins_1"] + feature_map["deg"]),
        ("KSE_SN1_Degree", feature_map["kse_0"] + feature_map["deg"]),
        ("KSE_SN2_Degree", feature_map["kse_1"] + feature_map["deg"]),
        ("Full", list(range(offset)) + feature_map["deg"])
     ]
        params = {
            "max_depth": 4,
            "eta": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_bin": 16,
            "scale_pos_weight": ratio_treino 
        }

        def evaluate(y_true, y_score, modelo_nome):
            pos_idx = np.where(y_true == 1)[0]
            neg_idx = np.where(y_true == 0)[0]

            for name, ratio in [("1:1", 1), ("1:10", 10), ("1:100", 100)]:
                n_neg = min(len(pos_idx) * ratio, len(neg_idx))
                neg_sample = rng.choice(neg_idx, n_neg, replace=False)
                idx = np.concatenate([pos_idx, neg_sample])

                yt = y_true[idx]
                ys = y_score[idx]

                auroc = roc_auc_score(yt, ys)
                aupr = average_precision_score(yt, ys)

                print(f"Rácio {name} -> AUROC: {auroc:.4f} | AUPR: {aupr:.4f} (N={len(idx)})")
                
                # === GUARDA OS RESULTADOS NA LISTA ===
                resultados_experimento.append({
                    "Dataset": args.dataset,
                    "Seed": args.CurrentSeed,
                    "SampleTecnic" : args.SampleTecnic,
                    "Features": modelo_nome,
                    "Ratio": name,
                    "AUROC": round(auroc, 4),
                    "AUPR": round(aupr, 4),
                    "N_Samples": len(idx)
                })

        for nome, feats in avaliacoes:
            print(f"\nTreinando: {nome}")

            Xtr = np.ascontiguousarray(X_train[:, feats], dtype=np.float32)
            Xte = np.ascontiguousarray(X_test[:, feats], dtype=np.float32)

            dtrain = xgb.DMatrix(Xtr, label=y_train)
            dtest = xgb.DMatrix(Xte)

            model = xgb.train(params, dtrain, num_boost_round=100)
            y_score = model.predict(dtest)
            print(y_score)
            print(f"Resultados: {nome}")
            
            evaluate(y_test, y_score, nome) 
            
            del Xtr, Xte, dtrain, dtest, model, y_score
            gc.collect()

        arquivo_resultados = os.path.join(args.exec_path, "data", "exp","resultados.csv")
    
        df_resultados = pd.DataFrame(resultados_experimento)
        
        arquivo_existe = os.path.isfile(arquivo_resultados)
        df_resultados.to_csv(arquivo_resultados, mode='a', index=False, header=not arquivo_existe)


        del X_train, X_test, y_train, y_test
        gc.collect()
def getL3(args, train_graph_path):
    # 1. Preparação dos dados
    train_graph_df = pd.read_csv(train_graph_path, header=None)
    train_graph_df[[0,1]] = train_graph_df[[1,0]]
    train_graph_df["weight"] = 1
    
    # 2. Definição dos caminhos
    l3_dir = f"{args.exec_path}/data/kpisti-L3-a163e9f"
    input_graph_path = f"{l3_dir}/graph.txt"
    output_graph_path = f"{l3_dir}/L3_predictions_graph.txt"
    cpp_source = f"{l3_dir}/L3.cpp"
    exec_path = f"{l3_dir}/teste.out"
    
    # 3. Limpeza de execuções anteriores
    if os.path.exists(input_graph_path): 
        os.remove(input_graph_path)
    if os.path.exists(output_graph_path): 
        os.remove(output_graph_path)
        
    train_graph_df.to_csv(input_graph_path, header=None, sep=" ", index=False)
    
    # ==========================================
    # 4. Compilação do código C++ (AGORA COM ASPAS NOS PATHS)
    # ==========================================
    # Coloquei aspas duplas ao redor do cpp_source e do exec_path
    compile_cmd = f'g++ "{cpp_source}" -o "{exec_path}" -O3'
    
    print("Compilando o código C++...")
    compilacao = subprocess.run(compile_cmd, shell=True, cwd=l3_dir, capture_output=True, text=True)
    
    if compilacao.returncode != 0:
        print(f"ERRO DE COMPILAÇÃO:\n{compilacao.stderr}")
        raise RuntimeError("Falha ao compilar o L3.cpp")
    # ==========================================
    
    # 5. Execução do C++ (TAMBÉM COM ASPAS)
    inicio = time.perf_counter()
    
    print("Executando a predição L3...")
    # Aspas duplas ao redor do executável e dos arquivos de entrada/saída
    run_cmd = f'"{exec_path}" "{input_graph_path}" "{output_graph_path}"'
    execucao = subprocess.run(run_cmd, shell=True, cwd=l3_dir, capture_output=True, text=True)
    
    if execucao.returncode != 0:
        print(f"ERRO NA EXECUÇÃO DO C++:\n{execucao.stderr}")
        raise RuntimeError("O executável do L3 falhou.")
        
    fim = time.perf_counter()
    total_time = fim - inicio
    
    # 6. Leitura dos resultados
    result = pd.read_csv(output_graph_path, sep="\t", header=None)
    result.columns = ["rightNode", "leftNode", "pred"]
    result["leftNode"] = result["leftNode"].astype(int)
    result["rightNode"] = result["rightNode"].astype(int)
 
    return [result[["leftNode", "rightNode", "pred"]], total_time]

def getGD(train_graph_path,
           max_dist: int = 3,
           alpha: float = 1.0,
           exclude_existing: bool = True,
           normalize: bool = False):
    
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

    adj = defaultdict(set)
    existing_edges = set()
    for ln, rn in zip(g["leftNode"].values, g["rightNode"].values):
        adj[ln].add(rn)
        adj[rn].add(ln)
        existing_edges.add((ln, rn))

    rows = []
    for ln in left_nodes:
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
                    
        for rn, d in dist.items():
            if rn in right_set and rn != ln and d >= 1 and (d % 2 == 1):
                if exclude_existing and (ln, rn) in existing_edges:
                    continue
                score = 1/d
                rows.append({"leftNode": ln, "rightNode": rn, "pred": score})

    fim = time.time()
    total_time = fim - inicio

    if not rows:
        return [pd.DataFrame(columns=["leftNode","rightNode", "pred"]), total_time]
    
    df_out = pd.DataFrame(rows)
    
    if normalize:
        minv = df_out["pred"].min()
        maxv = df_out["pred"].max()
        if maxv > minv:
            df_out["pred"] = (df_out["pred"] - minv) / (maxv - minv)

    df_out = df_out.sort_values("pred", ascending=False).reset_index(drop=True)
    df_out = df_out[["leftNode","rightNode", "pred"]]
    
    return [df_out, total_time]

def getKatz(train_graph_path,
            max_path_len: int = 5,
            beta: float = 0.05,
            symmetrize: bool = True,
            threshold: float = 1e-6): 
    """
    Compute Katz index scores memory-efficiently.
    """
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

    S = sp.csr_matrix((n, n), dtype=float)
    A_pow = A.copy()
    
    for l in range(1, max_path_len + 1):
        term = A_pow.copy()
        term.data *= (beta ** l)
        S = S + term
        
        if l < max_path_len:
            A_pow = A_pow.dot(A)
            
            A_pow.data[A_pow.data < threshold] = 0
            A_pow.eliminate_zeros()

    left_mask = np.isin(nodes_arr, left_list)
    right_mask = np.isin(nodes_arr, right_list)
    
    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]

    S_bip = S[left_idx, :][:, right_idx]
    S_coo = S_bip.tocoo()

    if S_coo.nnz == 0:
        return pd.DataFrame(columns=["leftNode", "rightNode", "pred"])
    fim = time.time()
    total_time = fim - inicio
    lefts = nodes_arr[left_idx[S_coo.row]]
    rights = nodes_arr[right_idx[S_coo.col]]
    preds = S_coo.data

    df_out = pd.DataFrame({"leftNode": lefts, "rightNode": rights, "pred": preds})

    df_out = df_out[df_out["leftNode"] != df_out["rightNode"]]

    df_out = df_out.sort_values("pred", ascending=False).reset_index(drop=True)

    return [df_out,total_time]



if __name__ == "__main__":
    args = overall_args()
    for seed in [0,1,2,3,4]:
        args.CurrentSeed = seed
        for filter in [0]:
            for dataset in ["ml-100k","citeulike","lastfm"]:
                for SampleTecnic in ["random"]:
                    for InvertedGraph in ["direct"]:
                        args.InvertedGraph = InvertedGraph
                        args.dataset = str(dataset)
                        args.Filter = filter
                        args.SampleTecnic = SampleTecnic
                        args.InvertedGraph = False
                        runMethod(args)
                        time.sleep(15)
                        # gc.collect()
                        # evaluateBaselines(args)
                        gc.collect()
                        time.sleep(15)

