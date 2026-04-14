import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pycatch22
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
from collections import defaultdict






def readFormatProbabilitys(adress:str):
    data = pd.read_csv(adress,header=None)

    print(data[2].sort_values())
    values = []
    for i,j in data.groupby(1):
        values.append([j[0].values[0],j[1].values[0],j[2].values])

    df = pd.DataFrame(values)
    df = df.rename(columns={0:"leftNode",
                       1:"rightNode",
                       2:"probability"})
    return df

def readFormatExcludedlinks(adress:str):

   
    dfExcluded = pd.read_csv(adress,header=None)
    dfExcluded.rename(columns={0:"leftNode",
                               1:"rightNode"},inplace=True)
    dfExcluded["leftNode"] = dfExcluded["leftNode"].astype("int64")
    
    return dfExcluded



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
    X = pd.DataFrame(ProbValues).reset_index(drop=True)
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


    for ind,i in enumerate(blue_idx):
        vals = X.iloc[i].dropna().values
        if vals.size > 0 and ind>1 :
            sns.histplot(math.sqrt(math.pow(float(np.log10(vals + 1)), 2) + math.pow(float(np.log10(vals + 1)), 2)), kde=True, element="step", palette=["blue", "red"],label="False edge")
            # sns.kdeplot(vals, fill=True, color="blue", label="Classe A", bw_adjust=0.5)
            # plt.hist(vals, bins=bins, density=True, color="blue")
            break
    
    for ind,i in enumerate(red_idx):
        vals = X.loc[i].dropna().values
        if vals.size > 0:
            sns.histplot(math.sqrt(math.pow(float(np.log10(vals + 1)), 2) + math.pow(float(np.log10(vals + 1)), 2)), kde=True, element="step", palette=["blue", "red"],label="Real edge")
            # sns.kdeplot(vals, fill=True, color="red", label="Classe B", bw_adjust=0.5)
            # plt.hist(vals, bins=bins, density=True, color="red")
            break

    # for ind,i in enumerate(green_idx):
        # vals = X.iloc[i].dropna().values
    #     if vals.size > 0 :
    #         plt.hist(vals, bins=bins, density=True, color="green", alpha=alpha, histtype='stepfilled', linewidth=lw)
    # legend proxies so legend isn't crowded
            # handles = [
            #     mpatches.Patch(color="green", label="label == 1 (masked)"),
            #     mpatches.Patch(color="blue",  label="label == 0 & original == 0"),
            #     mpatches.Patch(color="red",   label="original == 1")
            # ]
            # plt.legend(handles=handles)

    plt.xlabel("Value", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    # plt.yscale("log")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.title("Histogram of the embedding space", fontsize=18)
    plt.tight_layout()
    plt.show()

def evaluation(reverse = False):
    dfProbsIndex = readFormatProbabilitys("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/direct/1_13.csv")
    dfExcluded = readFormatExcludedlinks("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/testGraph.csv")
    OriginalEdges = pd.read_csv("/home/edu/Area_de_Trabalho/Projs/links/data/exp/ml-100k/normal/trainGraph.csv",header = None)
   
    OriginalEdges = OriginalEdges.rename(columns={1:"rightNode",
                       0:"leftNode"})
    
    OriginalEdges = OriginalEdges[["leftNode","rightNode"]]
    dfExcluded = dfExcluded[["leftNode","rightNode"]]

    if(reverse == True):
        dfExcluded[["rightNode","leftNode"]] = dfExcluded[["leftNode","rightNode"]]
        OriginalEdges[["rightNode","leftNode"]] = OriginalEdges[["LeftNode","rightNode"]]




    dfProbsValues = pd.DataFrame([i for i in dfProbsIndex.probability.values])

    

    dfProbsIndex["label"] = dfProbsIndex.rightNode.isin(dfExcluded[dfExcluded["leftNode"] == 13].rightNode.values).astype(int)
    
    excluded_rights = dfExcluded.loc[dfExcluded["leftNode"] == 13, "rightNode"].values
    train_rights = OriginalEdges.loc[OriginalEdges["leftNode"] == 13, "rightNode"].values
    dfProbsIndex["original"] = (dfProbsIndex["rightNode"].isin(train_rights) | dfProbsIndex["rightNode"].isin(excluded_rights)).astype(int)


    # for col in dfProbsValues.columns:
    #     media = dfProbsValues[col].mean()
    #     dfProbsValues.loc[dfProbsValues[col].isin([0,1]), col] = media
    # print(dfProbsValues[dfProbsValues[0] > 1])
    return dfProbsIndex,dfProbsValues

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



def plot2x2scatter(ProbsIndex,ProbValues,Pca):
    """
    Compute catch22 features for each time-series in ProbValues and plot a separate 2D scatter
    for every pair of catch22 features. Points are colored by 'original' (0=blue, 1=red)
    and masked points ((label==1) & (original==0)) are highlighted as black X.
    """
    import itertools



   # ProbValues = pd.DataFrame(ProbValues.rolling(window=100,min_periods=1).mean().values)

    # ensure ProbValues has time in rows and series in columns like in other functions
    ts = ProbValues.T

    # compute catch22 for each series (column)
    feats = [pycatch22.catch22_all(ts.iloc[:, i].values) for i in range(ts.shape[1])]
    if len(feats) == 0:
        return

    feat_names = feats[0]['names']
    feat_values = pd.DataFrame([f['values'] for f in feats], columns=feat_names)

    # align with ProbsIndex rows
    idx = ProbsIndex.reset_index(drop=True).index
    feat_values = feat_values.reset_index(drop=True)
    if len(feat_values) != len(ProbsIndex):
        # try to align by using min length
        n = min(len(feat_values), len(ProbsIndex))
        feat_values = feat_values.iloc[:n].reset_index(drop=True)
        meta = ProbsIndex.reset_index(drop=True).iloc[:n].reset_index(drop=True)
    else:
        meta = ProbsIndex.reset_index(drop=True)

    scaled_df = pd.concat([feat_values, meta[['label', 'original']].reset_index(drop=True)], axis=1)

    # plot each pair of features in its own figure
    for a, b in itertools.combinations(feat_names, 2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"] == 0)][a], scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"] == 0)][b],
                   c="blue", label="original=0", alpha=0.6, s=20)
        ax.scatter(scaled_df[scaled_df["original"] == 1][a], scaled_df[scaled_df["original"] == 1][b],
                   c="red", label="original=1", alpha=0.6, s=20)

        mask = (scaled_df["label"] == 1)
        if mask.any():
            ax.scatter(scaled_df.loc[mask, a], scaled_df.loc[mask, b],
                       c="black", marker="X", s=60, label="masked")

        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.legend()
        ax.set_title(f"{a} vs {b}")
        plt.tight_layout()
        plt.show()


def plotScatter(ProbsIndex,ProbValues):

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()

    pca = PCA(n_components=2)
    scaler = StandardScaler()
    # ProbValues = ProbValues.diff(axis=1).dropna(axis=1)
    X_scaled = scaler.fit_transform(ProbValues)
    values = pca.fit_transform(pd.DataFrame(X_scaled).fillna(0))
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    # values = reducer.fit_transform(X_scaled)
    scaled_df = pd.concat([pd.DataFrame(values), ProbsIndex[['label','original']].reset_index(drop=True)], axis=1)
    print(scaled_df)

    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label"])
    blue = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label"])
    # blue = blue.sample(200)
    red = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label"])

    # for col in red.columns:
    #     media = red[col].mean()
    #     red.loc[red[col].isin([1]), col] = media



    ax.scatter(x=blue[blue.columns[0]],y= blue[blue.columns[1]],c="blue")
    ax.scatter(x=red[blue.columns[0]],y= red[blue.columns[1]],c="red")
    ax.scatter(x=green[blue.columns[0]],y= green[blue.columns[1]],c="green")



    # print(green.shape)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Reversed Embedding space \n PCA plot", fontsize=20)
    plt.xlabel(f'Principal Component 1', fontsize=20)
    plt.ylabel(f'Principal Component 2', fontsize=20)



    plt.show()

def plotColumns(ProbsIndex,ProbValues):
    # scaler = StandardScaler()

    # X_scaled = scaler.fit_transform(ProbValues)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()

    # ProbValues = ProbValues.apply(lambda row: row.mask(row < 0.1, row.mean()), axis=1)

    scaled_df = pd.concat([ProbValues, ProbsIndex[['label','original']].reset_index(drop=True)], axis=1)
    green = scaled_df[scaled_df["label"] == 1].copy().drop(columns=["original","label"])
    blue = scaled_df[(scaled_df["label"] == 0) & (scaled_df["original"]==0)].copy().drop(columns=["original","label"])
    red = scaled_df[(scaled_df["original"] == 1)].copy().drop(columns=["original","label"])



    red["skew"] = np.log10(red.skew(axis=1))
    red["kurt"] = np.log10(red.kurt(axis=1))
    print(red.shape)

    # blue = blue.sample(4000)
    blue["skew"] = np.log10(blue.skew(axis=1)+1)
    blue["kurt"] = np.log10(blue.kurt(axis=1)+1)
    print(blue.shape)

    green["skew"] = np.log(green.skew(axis=1)+1)
    green["kurt"] = np.log10(green.kurt(axis=1)+1)
    print(green.shape)
    # ax.set_xlim(-1,1)
    print(red)
    ax.scatter(blue["kurt"],np.power(np.power(blue["skew"],2)+np.power(blue["skew"],2),0.5),c="blue",label="False egde")
    ax.scatter(red["kurt"],np.power(np.power(red["skew"],2)+np.power(red["kurt"],2),0.5),c="red",label = "Real edge")
    ax.scatter(green["kurt"],np.power(np.power(green["skew"],2)+np.power(green["kurt"],2),0.5),c="green",label="Masked edge")

    # print(blue)

    # for column in red.columns:

    #     plt.scatter(red[column].skew(),red[column].kurt(),c="red")
    #     plt.scatter(blue[column].skew(),blue[column].kurt(),c="blue")
    #     plt.scatter(green[column].skew(),green[column].kurt(),c="green")
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='lower right',fontsize=18)
    plt.title("third and fourth momentum \n from embedding space",fontsize=20)
    plt.xlabel("skew",fontsize=20)
    plt.ylabel("kurt",fontsize=20)
    plt.plot()

    plt.show()

a,b = evaluation()
# print(b)
# b,d = filter_by_fisher(b,a,top_k=150)
# a,b = balance_classes(a,b)
# print_closest_kurt_skew(a,b)
# plotColumns(a,b)
# plotScatter(a,b)
# plot2x2scatter(a,b,Pca=1)
# plotColumns(a,b)
plotHist(a,b)
# plotHist(b,a)
# knn(a,b1
# svm_classify(a,b)
# svm_classify(a,b,balance=True)
# Checar di

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