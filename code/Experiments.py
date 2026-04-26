from evaluation_v2 import evaluateAllModels, run_ablation_study,readFormatExcludedlinks
from create_graph import create_graph
import subprocess
import pandas as pd
from args import *
import sys
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time
import glob

def runMethod(args):
    if (args.CurrentSeed > 0):
        inicio = time.perf_counter()
        create_graph(args)
        
        # Diretório base para a semente atual
        base_saveFile = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0], args.SampleTecnic, args.InvertedGraph, str(args.CurrentSeed))
        codeFile = os.path.join("/home/edu/Area_de_Trabalho/Projs/LPv2/Code", "OurMethodExp.cpp")
        org = os.path.join("/home/edu/Area_de_Trabalho/Projs/LPv2/", "Code", "prob_calc.cpp")
        exe = os.path.join("/home/edu/Area_de_Trabalho/Projs/LPv2/Code", "build", "OurMethod")  
        os.makedirs(os.path.dirname(exe), exist_ok=True)

        # Compilar C++ apenas UMA VEZ (com flag -O3 para máxima performance!)
        print("Compilando C++...")
        compile_cmd = ["g++", "-O3", "-I", "/usr/include/eigen3", org, codeFile, "-o", exe]
        subprocess.run(compile_cmd, check=True)
        
        trainFile = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0], args.SampleTecnic, f"trainGraph{args.CurrentSeed}.csv")
        testFile = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0], args.SampleTecnic, f"testGraph{args.CurrentSeed}.csv")

        # ====================================================================
        # 1. AVALIAR BASELINES (Executa apenas uma vez)
        # ====================================================================
        # try:
        #     print("\n--- Avaliando Baselines (L3, CN, etc) ---")
        #     results_baselines = evaluateAllModels(args,
        #                     ourmethod_dir = base_saveFile,
        #                     train_graph_path = trainFile,
        #                     test_graph_path = testFile,
        #                     random_state = 0)
            
        #     linhas_formatadas = []
        #     for modelo, cenarios in results_baselines.items():
        #         for cenario, metricas in cenarios.items():
        #             if isinstance(metricas, dict):
        #                 linhas_formatadas.append({
        #                     "Model": modelo,
        #                     "Scenario": cenario,
        #                     "AUPR": metricas.get("aupr", 0),
        #                     "AUROC": metricas.get("auroc", 0),
        #                     "Time": metricas.get("time", 0), 
        #                 })
        #     if linhas_formatadas:
        #         df_resultados = pd.DataFrame(linhas_formatadas)
        #         df_resultados.to_csv(os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0], args.SampleTecnic, f"resultados{args.CurrentSeed}.csv"), index=False)
        # except Exception as e:
        #     print(f"Erro nos baselines: {str(e)}")
        #     traceback.print_exc(file=sys.stdout)

        # ====================================================================
        # 2. LOOP DAS 4 AVALIAÇÕES DO OURMETHOD
        # ====================================================================
        # Tuplos: (NeighSize, Mode, Descrição)

        resultados_finais_our = []

        for sn in [1, 2]:
            for mode in [0, 2]:
                print(f"\n{'='*50}")
                print(f"RODANDO EXPERIMENTO | SN: {sn} | MODE: {mode}")
                print(f"{'='*50}")
                
                # 1. Executa o C++
                run_cpp = [
                    exe, 
                    trainFile, 
                    base_saveFile,
                    str(sn), 
                    str(args.CurrentSeed),
                    str(mode), 
                ]
                subprocess.run(run_cpp, check=True)
                
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
                for file_path in glob.glob(os.path.join(base_saveFile, file_pattern)):
                    data = np.fromfile(file_path, dtype=dtype_our)
                    df_list.append(pd.DataFrame(data))

                if not df_list:
                    print(f"❌ Nenhum ficheiro encontrado em {base_saveFile} com o padrão {file_pattern}")
                    return {"full": {"aupr": 0, "auroc": 0}, "1:1": {"aupr": 0, "auroc": 0}, "1:10": {"aupr": 0, "auroc": 0}}

                df_preds = pd.concat(df_list, ignore_index=True)
                df_preds = df_preds.rename(columns={"col": "leftNode", "row1": "rightNode"})

                # --- 2. CARREGAR GROUND-TRUTH E MONTAR UNIVERSO ---
                train_raw = pd.read_csv(trainFile, header=None, names=["leftNode", "rightNode"]).astype(int)
                train_raw["is_train"] = 1
                train_raw["label"] = 1 

                df_test = readFormatExcludedlinks(testFile)
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
                df_merged["rightDegree"] = np.log1p(df_merged["rightNode"].map(right_degrees).fillna(0))

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
                df_train_neg = df_negatives_all.sample(n=num_train_pos, random_state=args.CurrentSeed)
                df_train_neg['is_train'] = 1
                
                df_test_neg_pool = df_negatives_all.drop(df_train_neg.index)
                df_test_neg_pool['is_train'] = 0

                # --- 5. TREINAR O CLASSIFICADOR (Random Forest) ---
                train_balanced = pd.concat([df_train_pos, df_train_neg])
                test_df = pd.concat([df_test_pos, df_test_neg_pool])
                print(train_balanced)



                try:
                    resultados = run_ablation_study(
                        train_balanced, 
                        test_df, 
                        sn=sn, 
                        mode=mode, 
                        random_seed=args.CurrentSeed
                    )
                    
                    # 3. Guarda os resultados para o CSV final
                    for res in resultados:
                        res["Dataset"] = args.dataset
                        resultados_finais_our.append(res)
                
                except Exception as e:
                    print(f"Erro na avaliação sn={sn} mode={mode}: {str(e)}")
                    traceback.print_exc()

        # Guarda os resultados conjuntos de OurMethod num só ficheiro CSV
        if resultados_finais_our:
            df_our = pd.DataFrame(resultados_finais_our)
            out_csv = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0], args.SampleTecnic, f"OurResultados{args.CurrentSeed}.csv")
            df_our.to_csv(out_csv, index=False)
            print(f"\n✅ Resultados OurMethod consolidados em: {out_csv}")

if __name__ == "__main__":
    args = overall_args()
    for seed in args.RandomSeed:
        args.CurrentSeed = seed
        for dataset in ["ml-100k", "yelp", "lastfm/user_artists"]:
            for SampleTecnic in ["normal"]:
                for InvertedGraph in ["direct"]:
                    args.InvertedGraph = InvertedGraph
                    args.dataset = str(dataset)
                    args.SampleTecnic = SampleTecnic
                    runMethod(args)