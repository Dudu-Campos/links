import pandas as pd
from args import *
import matplotlib.pyplot as plt


def read_all_csvs(adress):
    path = Path(f"{args.exec_path}/{adress}")
    dfs = []
    for file in path.iterdir():
        if file.suffix == ".csv":
            try:
                dfs.append(pd.read_csv(file,index_col=0,names = ["left_node","right_node"]))   
            except:
                print(f"arquivo {file} não conseguiu ser lido")
    df = pd.concat(dfs,axis=0)
    return df

def analyse_perturbation(data):
    print(data.head())
    plt.scatter(data.value_counts().values,[i for i in range(len(data.value_counts().values))])
    plt.xscale("log")
    plt.show()

if __name__ == "__main__":
    args = overall_args()
    df = read_all_csvs("data/movies/testes")
    analyse_perturbation(df)