import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
import pycatch22
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import os
import duckdb


from scipy.stats import skew, kurtosis


def readFormatExcludedlinks(adress:str):

   
    dfExcluded = pd.read_csv(adress,header=None)
    dfExcluded.rename(columns={0:"leftNode",
                               1:"rightNode"},inplace=True)
    dfExcluded["leftNode"] = dfExcluded["leftNode"].astype("int64")
    
    return dfExcluded

def plotHistogramKurtRatio(ProbsIndex, ProbValues):
    """
    Plot histograms of the kurt_ratio feature for each class.
    """
    # Combine ProbValues and ProbsIndex for easier processing
    scaled_df = pd.concat([ProbValues, ProbsIndex[['label', 'original']].reset_index(drop=True)], axis=1)

    # Separate classes
    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original", "label"])
    blue = scaled_df[(scaled_df["label"] == 0)].copy().drop(columns=["original", "label"])
    red = scaled_df[scaled_df["original"] == 1].copy().drop(columns=["original", "label"])

    # Calculate kurtosis and skewness
    green_kurt = green.kurt(axis=1)
    green_skew = green.skew(axis=1)
    green_kurt_ratio = green_kurt / (green_skew * green_skew)

    blue_kurt = blue.kurt(axis=1)
    blue_skew = blue.skew(axis=1)
    blue_kurt_ratio = blue_kurt / (blue_skew * blue_skew)

    red_kurt = red.kurt(axis=1)
    red_skew = red.skew(axis=1)
    red_kurt_ratio = red_kurt / (red_skew * red_skew)

    # Plot histograms
    plt.figure(figsize=(10, 6))
    sns.histplot(green_kurt_ratio, color="green", label="Masked edges", kde=True, bins=30,log_scale=(True, False))
    sns.histplot(blue_kurt_ratio, color="blue", label="False edges", kde=True, bins=30,log_scale=(True, False))
    sns.histplot(red_kurt_ratio, color="red", label="Real edges", kde=True, bins=30,log_scale=(True, False))

    plt.title("Histogram of the Normality feature by Class", fontsize=16)
    plt.xlim(0.25,100)
    plt.ylim(0.25,100)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def clean_anomalous_probs(df_probs):
    """
    Substitui probabilidades > 10 ou < -10 pela média da linha,
    garantindo que não distorçam a assimetria e curtose.
    """
    # 1. Cria uma máscara onde True significa "é uma anomalia"
    mask_anomalies = (df_probs > 10.0) | (df_probs < -10.0)
    
    # 2. Apaga temporariamente as anomalias e os zeros para não puxarem a média para baixo
    df_temp = df_probs.mask(mask_anomalies).replace(0.0, np.nan)
    
    # 3. Calcula a média real de caminhos válidos para cada linha
    row_means = df_temp.mean(axis=1)
    
    # 4. Substitui as anomalias pela média da respectiva linha
    # E onde a linha inteira era anômala/vazia (mean virou NaN), preenchemos com 0
    df_cleaned = df_probs.where(~mask_anomalies, row_means, axis=0).fillna(0.0)
    
    return df_cleaned


def plotHist(ProbsIndex,ProbValues):
    """
    Plot the distribution (histogram) of every row/series in ProbValues,
    coloring series by class:
      - green: (label == 1)
      - blue:  (label == 0) & (original == 0)
      - red:   (original == 1)

    Each series' histogram is plotted with low alpha so the aggregate shape is visible
    while still showing individual contributions.
    """
    import matplotlib.patches as mpatches

    # ensure DataFrame alignment
    ProbValues = ProbValues.fillna(0)
    X = pd.DataFrame(ProbValues).drop(columns=["leftNode","rightNode"]).reset_index(drop=True)
    meta = ProbsIndex.reset_index(drop=True)
    n = min(len(X), len(meta))
    if n == 0:
        print("No data to plot.")
        return
    X = X.iloc[:n]
    meta = meta.iloc[:n]

    plt.figure(figsize=(8, 6))
    bins = 50
    alpha = 0.05
    lw = 0.5

    # plot every series' histogram for each class with low alpha
    green_idx = meta.index[meta["label"] == 1].tolist()
    blue_idx = meta.index[(meta["label"] == 0) & (meta["original"] == 0)].tolist()
    red_idx = meta.index[meta["original"] == 1].tolist()

    blue_vals = X.iloc[blue_idx].mean(axis=1)
    red_vals  = X.iloc[red_idx].mean(axis=1)
    green_vals = X.iloc[green_idx].mean(axis=1)

    # remove NaN

    sns.kdeplot(blue_vals, color="blue", label="False", fill=True)
    sns.kdeplot(red_vals, color="red", label="Real", fill=True)
    sns.kdeplot(green_vals, color="green", label="Masked", fill=True)


    plt.legend()

    plt.xlabel("Value", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    # plt.yscale("log")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.title("Histogram of the embedding space", fontsize=18)
    plt.tight_layout()
    plt.show()

dtype = np.dtype([
    ("col", np.int32),
    ("row1", np.int32),
    ("prob", np.float32)
])


def plotRF(ProbsIndex, ProbValues):
    print(ProbsIndex.head())
    ProbValues = ProbValues.fillna(0)
    ProbValues_mean = ProbValues.mean(axis=1)
    ProbValues_std = ProbValues.std(axis=1)
    ProbValues_skew = ProbValues.skew(axis=1)
    ProbValues_kurt = ProbValues.kurt(axis=1)
    ProbValues = ProbValues[["leftNode","rightNode"]]
    ProbValues = pd.concat([ProbValues[["leftNode","rightNode"]],pd.DataFrame([ProbValues_kurt,ProbValues_mean,ProbValues_std,ProbValues_skew]).T],axis=1)
    print(ProbValues)
    scaled_df = ProbValues.merge(
        ProbsIndex[['leftNode', 'rightNode', 'label', 'original']],
        on=['leftNode', 'rightNode'],
        how='left'
    )

    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label","leftNode","rightNode"])
    blue  = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label","leftNode","rightNode"])
    red   = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label","leftNode","rightNode"])

    X_train = pd.concat([red, blue]).fillna(0)
    y_train = [1]*len(red) + [0]*len(blue)

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    X_test = pd.concat([green, blue]).fillna(0)
    y_test = [1]*len(green) + [0]*len(blue)
    scores = clf.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, scores)
    aupr  = average_precision_score(y_test, scores)
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR : {aupr:.4f}")

    # plot dos scores como distribuição 1D
    X_all    = pd.concat([red, blue, green]).fillna(0)
    scores_all = clf.predict_proba(X_all)[:, 1]
    n_red, n_blue = len(red), len(blue)

    plt.figure(figsize=(10,5))
    sns.kdeplot(scores_all[:n_red],              color="red",   label="Real edge",   fill=True, alpha=0.4)
    sns.kdeplot(scores_all[n_red:n_red+n_blue],  color="blue",  label="False edge",  fill=True, alpha=0.4)
    sns.kdeplot(scores_all[n_red+n_blue:],       color="green", label="Masked edge", fill=True, alpha=0.4)
    plt.legend()
    plt.title(f"Random Forest scores | AUROC={auroc:.3f} AUPR={aupr:.3f}")
    plt.show()

    return auroc, aupr

def readDataDB(Id,reverse = False):
    all_data = []
    con = duckdb.connect("/home/edu/Area_de_Trabalho/Projs/links/banco_grafos.duckdb")
    con.execute("CHECKPOINT;") # Força a consolidação dos dados pendentes
    res = con.execute("SELECT COUNT(*) FROM probabilidades").df()

    query = """
        SELECT col, row, prob 
        FROM probabilidades 
        WHERE col = 1 
        ORDER BY prob DESC
    """

    df_no = con.execute(query).df()
    print(df_no)
    all_data = []
    for i,j in df_no.groupby("row"):
        all_data.append(pd.DataFrame({"col":Id,
                                    "row1":j["row"].values[0],
                                      "prob":[j["prob"].values]}))


    con.close()

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)



    dfProbsIndex = combined_data[["col", "row1"]].copy()
    dfProbsIndex = dfProbsIndex.rename(columns={"col": "leftNode", "row1": "rightNode"})
    dfProbsIndex = dfProbsIndex.drop_duplicates() #
    # Create dfProbsValues
    combined_data = combined_data[(combined_data["col"] == Id)]

    df_expandido = pd.DataFrame(combined_data["prob"].to_list())

    
    df_expandido.columns = [f'probs_{i+1}' for i in df_expandido.columns]

    # Junta com o DataFrame original
    dfProbsValues = combined_data.drop('prob', axis=1).join(df_expandido)
    dfProbsValues = dfProbsValues.fillna(0)

    threshold = 0.80  # Dropa se a coluna for composta por mais de 80% de "vazios"
    
    # Cria uma máscara onde o valor é válido (não é NaN E não é Zero)
    is_valid = dfProbsValues.notna() & (dfProbsValues != 0)
    
    # Calcula a fração de valores válidos em cada coluna
    valid_frac = is_valid.mean(axis=0)
    # Mantém apenas colunas que possuem mais valores válidos do que o limite mínimo (ex: > 20%)
    cols_keep = valid_frac[valid_frac > (1 - threshold)].index
    dfProbsValues = dfProbsValues[cols_keep]   

    # 2. FAZER A IMPUTAÇÃO APENAS NAS COLUNAS SOBREVIVENTES
    dfProbsValues = dfProbsValues.rename(columns={"col": "leftNode", 
                                                   "row1": "rightNode"})
    

    # Add labels and original columns
    dfExcluded = readFormatExcludedlinks("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/testGraph1.csv")
    OriginalEdges = pd.read_csv("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/trainGraph1.csv", header=None)
    OriginalEdges = OriginalEdges.rename(columns={0: "leftNode", 1: "rightNode"})

    excluded_rights = dfExcluded.loc[dfExcluded["leftNode"] == Id, "rightNode"].values
    train_rights = OriginalEdges.loc[OriginalEdges["leftNode"] == Id, "rightNode"].values

    dfProbsIndex["label"] = dfProbsIndex["rightNode"].isin(excluded_rights).astype(int)
    dfProbsIndex["original"] = dfProbsIndex["rightNode"].isin(train_rights).astype(int)
    dfProbsIndex = dfProbsIndex[(dfProbsIndex["leftNode"] == Id)]

 
    return dfProbsIndex, dfProbsValues

def readData(Id,dir,reverse = True):
    all_data = []

    # Iterate over all files in the directory
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if file_name.endswith(".bin"):
            data = pd.DataFrame(np.fromfile(os.path.join(file_path), dtype=dtype))
            all_data.append(data)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Extract relevant columns
    # Create dfProbsIndex
    dfProbsIndex = combined_data[["col", "row1"]].copy()
    dfProbsIndex = dfProbsIndex.rename(columns={"col": "leftNode", "row1": "rightNode"})
    # if reverse:
    #     dfProbsIndex[["leftNode", "rightNode"]] = dfProbsIndex[["rightNode", "leftNode"]]
    dfProbsIndex = dfProbsIndex.drop_duplicates() #
    # Create dfProbsValues
    combined_data = combined_data[(combined_data["col"] == Id)]
    values = []
    for i, j in combined_data.groupby(["col", "row1"]):
        probs = j["prob"].tolist() 
        values.append([i[0], i[1]] + probs)
        
    dfProbsValues = pd.DataFrame(values)
    dfProbsValues = dfProbsValues.rename(columns={0: "leftNode", 
                                                   1: "rightNode",
                                                   2:"Probability"})
    prob_cols = {i: f"prob_{i-2}" for i in range(2, dfProbsValues.shape[1])}
    # dfProbsValues = dfProbsValues.fillna(0)
    dfProbsValues.rename(columns=prob_cols, inplace=True)
    # Add labels and original columns
    dfExcluded = readFormatExcludedlinks("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/testGraph0.csv")

    OriginalEdges = pd.read_csv("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/trainGraph0.csv", header=None)
    OriginalEdges = OriginalEdges.rename(columns={0: "leftNode", 1: "rightNode"})

    excluded_rights = dfExcluded.loc[dfExcluded["leftNode"] == Id, "rightNode"].values
    train_rights = OriginalEdges.loc[OriginalEdges["leftNode"] == Id, "rightNode"].values

    dfProbsIndex["label"] = dfProbsIndex["rightNode"].isin(excluded_rights).astype(int)
    dfProbsIndex["original"] = dfProbsIndex["rightNode"].isin(train_rights).astype(int)
    dfProbsIndex = dfProbsIndex[(dfProbsIndex["leftNode"] == Id)]

    return dfProbsIndex, dfProbsValues
# def normalizeProbs(ProbsIndex,ProbValues)


def plotTime(ProbsIndex,ProbValues):


    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()


    # ProbValues[(ProbsIndex.label ==1).index]
    # for k in ProbValues[(ProbsIndex.original ==0)].index[4:6]:
    #     plt.scatter([i for i in range(len(ProbValues.iloc[k,:]))],ProbValues.iloc[k,:].rolling(window=10).mean().values,label="existing outlier edge embeddings",c="green")
    #     plt.plot(ProbValues.iloc[k,:].rolling(window=10).mean().values,c="red")

    for k in ProbValues[(ProbsIndex.original == 1)].index[5:7]:
        ax.scatter([i for i in range(len(ProbValues.iloc[k,:]))],ProbValues.iloc[k,:].rolling(window=10).mean().values,label="existing outlier edge embeddings",c="red")
        ax.plot(ProbValues.iloc[k,:].rolling(window=10).mean().values,c="red")

    for k in ProbValues[(ProbsIndex.original == 0) & (ProbsIndex.label ==0)].index[5:7]:
        ax.scatter([i for i in range(len(ProbValues.iloc[k,:]))],ProbValues.iloc[k,:].rolling(window=10).mean().values,label="existing outlier edge embeddings",c="blue")
        ax.plot(ProbValues.iloc[k,:].rolling(window=10).mean().values,c="blue")

    # plt.scatter([i for i in range(len(ProbValues.iloc[0,:]))],ProbValues.iloc[0,:].rolling(window=10).mean().values,label="non-existing outlier edge embeddings",c="blue")
    # plt.plot(ProbValues.iloc[0,:].rolling(window=10).mean().values,c="blue")

    # plt.scatter([i for i in range(len(ProbValues.iloc[151,:]))],ProbValues.iloc[151,:].rolling(window=10).mean().values,label="existing edge embeddings",c="red")
    # plt.plot(ProbValues.iloc[151,:].rolling(window=10).mean().values,c="red")

    # plt.scatter([i for i in range(len(ProbValues.iloc[3,:]))],ProbValues.iloc[3,:].rolling(window=10).mean().values,label="non-existing edge embeddings",c="red")
    # plt.plot(ProbValues.iloc[3,:].rolling(window=10).mean().values,c="red")

    plt.show()





def plotScatter(ProbsIndex,ProbValues):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    pca = PCA(n_components=2)

    print(ProbValues)
    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(df)

    # pca = PCA(n_components=2) # Specify number of components or % variance (e.g., 0.95)
    # pca.fit(df_scaled)
    # df_pca = pd.DataFrame(pca.transform(df_scaled))
    # print(df_pca[0].unique)
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    # values = reducer.fit_transform(X_scaled)
    scaled_df = pd.concat([ProbValues.drop(columns=["leftNode","rightNode"]), ProbsIndex[['label','original']].reset_index(drop=True)], axis=1)
    print(scaled_df)

    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label"])
    blue = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label"])
    # blue = blue.sample(200)
    red = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label"])

    red_mean = red.sum(axis=1)
    red_std = red.std(axis=1)
    red["skew"] = red.skew(axis=1)
    red["kurt"] = red.kurt(axis=1)

    blue_mean = blue.sum(axis=1)
    blue_std = blue.std(axis=1)
    blue["skew"] = blue.skew(axis=1)
    blue["kurt"] = blue.kurt(axis=1)
   
    green_mean = green.sum(axis=1)
    green_std = green.std(axis=1)
    green["skew"] = green.skew(axis=1)
    green["kurt"] = green.kurt(axis=1)

    # ax.scatter(red_mean,red_std,color="red")
    # ax.scatter(blue_mean,blue_std,color="blue")
    # ax.scatter(green_mean,green_std,color="green")



    ax.scatter(red["skew"],red["kurt"],color="red")
    ax.scatter(blue["skew"],blue["kurt"],color="blue")
    ax.scatter(green["skew"],green["kurt"],color="green")

    print(green.shape)
    print(red.shape)
    print(blue.shape)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Reversed Embedding space \n PCA plot", fontsize=20)
    plt.xlabel(f'Principal Component 1', fontsize=20)
    plt.ylabel(f'Principal Component 2', fontsize=20)
    plt.show()


def plotScatter2(ProbsIndex, ProbValues):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    ProbValues = ProbValues.fillna(0)
    # 1. Isola apenas as colunas de valores (removendo nós e possíveis labels)
    ProbValues_mean = ProbValues.mean(axis=1)
    ProbValues_std = ProbValues.std(axis=1)
    ProbValues_skew = ProbValues.skew(axis=1)
    ProbValues_kurt = ProbValues.kurt(axis=1)
    ProbValues = pd.concat([ProbValues[["leftNode","rightNode"]],pd.DataFrame([ProbValues_kurt,ProbValues_mean,ProbValues_std,ProbValues_skew]).T],axis=1)

    features_df = ProbValues.drop(columns=["leftNode", "rightNode"], errors='ignore')
    
    # 2. Calcula os percentis por linha (axis=1). 
    # Os valores [0.9, 0.75, 0.5, 0.25, 0.1] representam p90, p75, p50, p25, p10
    # percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    # df_quantiles = features_df.quantile(percentiles, axis=1).T
    # df_quantiles.columns = [f'p{int(q*100)}' for q in percentiles]
    # df = pd.DataFrame({
    #     'P10': df_quantiles["p10"],
    #     'P25': df_quantiles["p25"],
    #     'P50': df_quantiles["p50"],
    #     'P75': df_quantiles["p75"],
    #     'P90': df_quantiles["p90"],
    #     'label': ProbsIndex['label'].reset_index(drop=True),
    #     'original': ProbsIndex['original'].reset_index(drop=True)
    # })
    # print(df.sort_values(by="P90").tail(20))
    # 3. Padroniza os dados calculados (Importante para o PCA)
    # scaler = StandardScaler()
    # quantiles_scaled = scaler.fit_transform(features_df)
    
    # # 4. Aplica o PCA com 2 componentes
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(quantiles_scaled)
    
    # 5. Junta os resultados do PCA com os labels para facilitar a separação
    scaled_df = pd.DataFrame({
        'PC1': ProbValues_mean+ProbValues_std,
        'PC2': ProbValues_kurt*ProbValues_skew,
        'label': ProbsIndex['label'].reset_index(drop=True),
        'original': ProbsIndex['original'].reset_index(drop=True)
    })
    print(scaled_df[scaled_df["PC2"] < 300])
    # 6. Separa os grupos conforme sua lógica original
    green = scaled_df[scaled_df["label"] == 1]
    blue  = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"] == 0)]
    red   = scaled_df[(scaled_df["original"] == 1)]

    
    ax.scatter(red['PC1'], red['PC2'], color="red", label='Original=1')
    ax.scatter(blue['PC1'], blue['PC2'], color="blue", label='Label=0 & Original=0')
    ax.scatter(green['PC1'], green['PC2'], color="black", label='Label=1')

    # Imprime os tamanhos para conferência
    print(f"Green shape: {green.shape}")
    print(f"Red shape: {red.shape}")
    print(f"Blue shape: {blue.shape}")
    
    # Customização do Plot
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("PCA do Espaço de Embedding \n(Baseado nos Percentis: p90, p75, p50...)", fontsize=18)
    
    # Mostra a variância explicada em cada eixo (opcional, mas muito informativo no PCA)
    # plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=15)
    # plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=15)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()





def plotBoxplotKurtRatio(ProbsIndex, ProbValues):
    """
    Create a boxplot for the kurt_ratio feature for each class.
    """
    # Combine ProbValues and ProbsIndex for easier processing
    scaled_df = pd.concat([ProbValues, ProbsIndex[['label', 'original']].reset_index(drop=True)], axis=1)

    # Separate classes
    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original", "label"])
    blue = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"] == 0)].copy().drop(columns=["original", "label"])
    red = scaled_df[scaled_df["original"] == 1].copy().drop(columns=["original", "label"])

    # Calculate kurtosis and skewness
    green_kurt = green.kurt(axis=1)
    green_skew = green.skew(axis=1)
    green_kurt_ratio = green_kurt / (green_skew * green_skew)

    blue_kurt = blue.kurt(axis=1)
    blue_skew = blue.skew(axis=1)
    blue_kurt_ratio = blue_kurt / (blue_skew * blue_skew)

    red_kurt = red.kurt(axis=1)
    red_skew = red.skew(axis=1)
    red_kurt_ratio = red_kurt / (red_skew * red_skew)

    # Combine data for boxplot
    data = {
        "Masked edges (green)": green_kurt_ratio,
        "False edges (blue)": blue_kurt_ratio,
        "Real edges (red)": red_kurt_ratio
    }

    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pd.DataFrame(data))
    plt.ylabel("Kurtosis Ratio", fontsize=14)
    plt.title("Boxplot of Kurtosis Ratio by Class", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import mahalanobis

def biased_kurtosis(df: pd.DataFrame) -> pd.Series:
    """µb(4, φ, e) — 4º momento centralizado no skew"""
    skew_vals = df.skew(axis=1)
    
    centered = df.sub(skew_vals, axis=0)  # (f_j - skew) para cada linha
    
    n = df.shape[1]
    mu4_skew = (centered ** 4).sum(axis=1) / n
    mu3_skew = (centered ** 3).sum(axis=1) / n
    
    return mu4_skew / (mu3_skew ** 2)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotLDA(ProbsIndex, ProbValues):

    ProbValues = ProbValues.fillna(0)
    scaled_df = ProbValues.merge(
        ProbsIndex[['leftNode', 'rightNode', 'label', 'original']],
        on=['leftNode', 'rightNode'],
        how='left'
    )
    x = ProbValues.copy()

 

    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label","leftNode","rightNode"])
    blue  = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label","leftNode","rightNode"])
    red   = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label","leftNode","rightNode"])

    # =========================
    # TREINO LDA
    # =========================
    X_train = pd.concat([red, blue]).fillna(0)
    y_train = [1]*len(red) + [0]*len(blue)  # 1=real, 0=false

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_train, y_train)

    # =========================
    # SCORES PARA AVALIAÇÃO
    # =========================
    X_test = pd.concat([green, blue])
    y_test = [1]*len(green) + [0]*len(blue)


    # score contínuo (melhor usar isso)
    scores = lda.decision_function(X_test)

    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(y_test, scores))
    # =========================
    # MÉTRICAS
    # =========================
    auroc = roc_auc_score(y_test, scores)
    aupr  = average_precision_score(y_test, scores)

    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR : {aupr:.4f}")

    # =========================
    # PROJEÇÃO PARA PLOT
    # =========================
    X_all = pd.concat([red, blue, green]).fillna(0)
    values = lda.transform(X_all).flatten()

    n_red  = len(red)
    n_blue = len(blue)

    plt.figure(figsize=(10,5))
    sns.kdeplot(values[:n_red],             color="red",   label="Real edge",   fill=True, alpha=0.4)
    sns.kdeplot(values[n_red:n_red+n_blue], color="blue",  label="False edge",  fill=True, alpha=0.4)
    sns.kdeplot(values[n_red+n_blue:],      color="green", label="Masked edge", fill=True, alpha=0.4)

    plt.legend()
    plt.title(f"LDA projection | AUROC={auroc:.3f} AUPR={aupr:.3f}")
    plt.show()

    return auroc, aupr

def add_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
    percentiles = [10, 25, 50, 75, 90]
    result = pd.DataFrame(index=df.index)
    
    for p in percentiles:
        result[f"p{p}"] = np.nanpercentile(df.values, p, axis=1)
    
    return result

from scipy.stats import mannwhitneyu
def select_discriminative_cols(red_df, blue_df, top_k=50):
    pvalues = {}
    
    for col in red_df.columns:
        r = red_df[col].dropna()
        b = blue_df[col].dropna()
        if len(r) > 0 and len(b) > 0:
            _, p = mannwhitneyu(r, b, alternative='two-sided')
            pvalues[col] = p
    
    pval_series = pd.Series(pvalues).sort_values()
    selected = pval_series.head(top_k).index.tolist()
    
    return selected, pval_series

def plotFrenqs(ProbsIndex,ProbValues):
    scaled_df = ProbValues.merge(
            ProbsIndex[['leftNode', 'rightNode', 'label', 'original']],
            on=['leftNode', 'rightNode'],
            how='left'
        )
    print(scaled_df.head())
    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label","leftNode","rightNode"])
    blue = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label","leftNode","rightNode"])
    red = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label","leftNode","rightNode"])

    x,y = select_discriminative_cols(red,blue,top_k=50)
    plt.hist(y)
    plt.show()


def plotColumns(ProbsIndex, ProbValues):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    # ProbValues = ProbValues.fillna(0)
    
    print(ProbsIndex)
    print(ProbValues.isna().sum().sum())
    
    # Fit and transform the data
    scaled_df = ProbValues.merge(
            ProbsIndex[['leftNode', 'rightNode', 'label', 'original']],
            on=['leftNode', 'rightNode'],
            how='left'
        )
    
    print(scaled_df.head())
    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label","leftNode","rightNode"])
    blue = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label","leftNode","rightNode"])
    red = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label","leftNode","rightNode"])

    # --- Red Calculations ---
    red_sum = red.mean(axis=1)
    red["skew"] = red.skew(axis=1)
    red["kurt"] = red.kurt(axis=1)
    red_filtered = red.drop(columns=["kurt", "skew"])
    red_probs = red_filtered.div(red_filtered.sum(axis=1), axis=0)
    red_entropy = red_probs.apply(lambda row: entropy(row, base=2), axis=1)

    # --- Blue Calculations ---
    blue_sum = blue.mean(axis=1)
    blue["skew"] = blue.skew(axis=1)
    blue["kurt"] = blue.kurt(axis=1)
    blue_filtered = blue.drop(columns=["kurt", "skew"])
    blue_probs = blue_filtered.div(blue_filtered.sum(axis=1), axis=0)
    blue_entropy = blue_probs.apply(lambda row: entropy(row, base=2), axis=1)

    # --- Green Calculations ---
    green_sum = green.mean(axis=1)
    green["skew"] = green.skew(axis=1)
    green["kurt"] = green.kurt(axis=1)
    green_filtered = green.drop(columns=["kurt", "skew"])
    green_probs = green_filtered.div(green_filtered.sum(axis=1), axis=0)
    green_entropy = green_probs.apply(lambda row: entropy(row, base=2), axis=1)

    # ==========================================
    # CÁLCULO DA DISTÂNCIA NORMALIZADA
    # ==========================================
    
    # 1. Preparar dados para normalização conjunta
    df_blue = pd.DataFrame({'x': blue["kurt"]*blue["skew"], 'y': blue_entropy, 'color': 'blue'})
    df_red = pd.DataFrame({'x': red["kurt"]*red["skew"], 'y': red_entropy, 'color': 'red'})
    df_green = pd.DataFrame({'x': green["kurt"]*green["skew"], 'y': green_entropy, 'color': 'green'})
    
    df_all = pd.concat([df_blue, df_red, df_green])
    
    # 2. Normalizar as variáveis X e Y de 0 a 1
    scaler = MinMaxScaler()
    df_all[['x_norm', 'y_norm']] = scaler.fit_transform(df_all[['x', 'y']])
    
    # 3. Calcular a Distância Euclidiana até o ponto (0, 1) no plano normalizado
    # (0 no eixo X normalizado e 1 no eixo Y normalizado, que equivale ao max(entropy))
    df_all['norm_distance'] = np.sqrt((df_all['x_norm'] - 0)**2 + (df_all['y_norm'] - 1)**2)

    # ==========================================
    # PLOTAGEM
    # ==========================================
    
    plt.xlabel("kurt * skew (Original)")
    plt.ylabel("Distância Normalizada até (0, Max Entropia)")
    
    # Plotando os dados usando o X original contra a nova distância
    blue_plot = df_all[df_all['color'] == 'blue']
    red_plot = df_all[df_all['color'] == 'red']
    green_plot = df_all[df_all['color'] == 'green']

    ax.scatter(blue_plot['x'], blue_plot['norm_distance'], color="blue", label="False", alpha=0.7)
    ax.scatter(red_plot['x'], red_plot['norm_distance'], color="red", label="Real", alpha=0.7)
    ax.scatter(green_plot['x'], green_plot['norm_distance'], color="green", label="Masked", alpha=0.7)
    # ax.scatter(blue_plot['x_norm'], blue_plot['y_norm'], color="blue", label="False", alpha=0.7)
    # ax.scatter(red_plot['x_norm'], red_plot['y_norm'], color="red", label="Real", alpha=0.7)
    # ax.scatter(green_plot['x_norm'], green_plot['y_norm'], color="green", label="Masked", alpha=0.7)
    
    plt.legend()

    # Mantendo seu print final
    blue_plot["color"] = "blue"
    red_plot["color"] = "green"
    green_plot["color"] = "red"
    final = pd.concat([red_plot, blue_plot, green_plot])
    print(final.sort_values(by="norm_distance").head(20))

    print(final.sort_values(by="norm_distance").head(50).color.value_counts())
    
    plt.show()


def plotmean(ProbsIndex,ProbValues):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()

    # Initialize scaler
    ProbValues = ProbValues.fillna(0)
    # Fit and transform the data
    scaled_df = ProbValues.merge(
            ProbsIndex[['leftNode', 'rightNode', 'label', 'original']],
            on=['leftNode', 'rightNode'],
            how='left'
        )
    
    print(scaled_df.head())
    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label","leftNode","rightNode"])
    blue = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label","leftNode","rightNode"])
    red = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label","leftNode","rightNode"])


    red_mean = red.sum(axis=1)
    red_var = red.skew(axis=1)


    blue_mean = blue.sum(axis=1)
    blue_var = blue.skew(axis=1)


    green_mean = green.sum(axis=1)
    green_var = green.skew(axis=1)





    plt.xlabel("kurt")
    plt.ylabel("skew")


    ax.scatter(blue["kurt"]*blue["skew"],blue.quantile(90,axis=1),color="blue")
    ax.scatter(red["kurt"]*red["skew"],red.quantile(90,axis=1),color="red")
    ax.scatter(green["kurt"]*green["skew"],green.quantile(90,axis=1),color="green")

    plt.show()


dtype = np.dtype([
    ("col", np.int32),
    ("row1", np.int32),
    ("kurt_skew", np.float32),
    ("entropy", np.float32),

])


# result = pd.DataFrame(np.fromfile(os.path.join("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/direct/0/final_kurtosis_output.bin"), dtype=dtype,))

def readSumarizedData(path):
    result = pd.DataFrame(np.fromfile(path), dtype=dtype,)
    return result

# result = pd.DataFrame(np.fromfile(os.path.join("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/direct/0/final_kurtosis_output.bin"), dtype=dtype,))


def classify_and_evaluate(ProbsIndex, ProbValues):
    # 1. Tratar nulos e juntar os DataFrames
    ProbValues = ProbValues.fillna(0)
    df_merged = ProbValues.merge(
        ProbsIndex[['leftNode', 'rightNode', 'label', 'original']],
        on=['leftNode', 'rightNode'],
        how='left'
    )
    
    # 2. Isolar apenas as colunas de probabilidade (nossa matriz bruta)
    prob_cols = [col for col in df_merged.columns if col not in ['leftNode', 'rightNode', 'label', 'original']]
    df_probs_only = df_merged[prob_cols]
    
    print("Extraindo features estatísticas (Skewness e Kurtosis)...")
    
    # 3. Calcular Skew e Kurtosis ao longo das colunas (axis=1) para formar o X
    X = pd.DataFrame({
        'kurtosis': df_probs_only.kurt(axis=1),
        'skewness': df_probs_only.skew(axis=1)
    }).fillna(0) # fillna(0) previne erros matemáticos caso a linha tenha variância zero
    
    # 4. Definir o alvo (y): 1 para arestas reais (originais ou ocultas), 0 para ruído
    y = df_merged['label']
    
    # 5. Dividir em conjunto de Treino e Teste (70% treino, 30% teste)
    # stratify=y garante que a proporção de arestas falsas/reais se mantenha igual
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Treinando o Random Forest Classifier...")
    
    # 6. Inicializar e treinar o classificador
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 7. Prever probabilidades no conjunto de teste
    # Pegamos a coluna [:, 1] que representa a probabilidade de ser classe 1 (Aresta Real)
    y_pred_probs = clf.predict_proba(X_test)[:, 1]
    
    # 8. Calcular as métricas
    auroc = roc_auc_score(y_test, y_pred_probs)
    aupr = average_precision_score(y_test, y_pred_probs)
    
    # 9. Exibir o Relatório
    print("\n========================================================")
    print(" RESULTADOS DO CLASSIFICADOR (Features: Skew + Kurt)")
    print("========================================================")
    print(f"AUROC (Área sob a curva ROC) : {auroc:.4f}")
    print(f"AUPR  (Área sob a curva PR)  : {aupr:.4f}")
    print("========================================================\n")
    
    return clf, auroc, aupr






import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


a,b = readDataDB(Id=1)
# plotScatter2(a,b)
plotColumns(a,b)
# plotHist(a,b)
# plotColumns(a,b)
# plotRF(a,b)
# plotScatter(a,b)
# plotHist(a,b)
# plotColumns(a,b)
# plotRF(a,b)
# plotScatter(a,b)
# classify_and_evaluate(a,b)
# plotScatter(a,b)
# print(b)
# b,d = filter_by_fisher(b,a,top_k=150)
# a,b = balance_classes(a,b)
# plotLDA(a,b)
# evaluateTopologicalSpace(a,b)
# plotHistogramKurtRatio(a, b)
# plotBoxplotKurtRatio(a,b)
# plotTrueCoupledMoments(a, b)
# train = pd.read_csv("/home/edu/Area_de_Trabalho/Projs/links/data/exp/movies/normal/trainGraph.csv", header=None)
# print("Grau médio left:", train.groupby(0).size().mean())
# print("Grau médio right:", train.groupby(1).size().mean())

# # checar se os positivos do teste são "óbvios" pelo grau
# test = pd.read_csv("/home/edu/Area_de_Trabalho/Projs/links/data/exp/movies/normal/testGraph.csv", header=None)
# test.columns = ["leftNode", "rightNode"]
# test_degrees = test.merge(train.rename(columns={0:"leftNode",1:"rightNode"})\
#                 .groupby("leftNode").size().reset_index(name="deg"),
#                 on="leftNode")
# print("Grau médio dos left nodes com arestas no teste:")
# print(test_degrees["deg"].describe())