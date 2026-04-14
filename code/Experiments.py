from evaluation import evaluateAllModels,evaluateOurknn
from create_graph import create_graph
import subprocess
import pandas as pd
from args import *
import os

def runMethod(args):
    create_graph(args)
    saveFile = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0],args.SampleTecnic,args.InvertedGraph)
    codeFile = os.path.join(args.exec_path, "code", "OurMehotdExp.cpp")
    org = os.path.join(args.exec_path, "code", "prob_calc.cpp")
    exe = os.path.join(args.exec_path, "build", "OurMethod")  
    os.makedirs(os.path.dirname(exe), exist_ok=True)

    # compile
    compile_cmd = ["g++", "-I", "/usr/include/eigen3",org, codeFile, "-o", exe]
    subprocess.run(compile_cmd, check=True)
    # run compiled program with arguments
    trainFile =  os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0],args.SampleTecnic,"trainGraph.csv")
    run_cmd = [exe, trainFile, saveFile, str(args.NeighSize), str(args.PruningRatio),str(args.InvertedGraph)]

    subprocess.run(run_cmd, check=True)

                     #  ourmethod_dir = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0],args.SampleTecnic,args.InvertedGraph),
    # results = evaluateOurknn(ourmethod_dir =  os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0],args.SampleTecnic,args.InvertedGraph),
    #                 train_graph_path = os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0],args.SampleTecnic,"trainGraph.csv"),
    #                 test_graph_path =  os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0],args.SampleTecnic,"testGraph.csv"),
    #                 katz_max_path_len = 5,
    #                 katz_beta = 0.05,
    #                 random_state = args.RandomSeed,
    #                 verbose = True)
    
    # results = pd.DataFrame(results)
    # results.to_csv(os.path.join(args.exec_path, "data", "exp", args.dataset.split("/")[0],args.SampleTecnic,"resultados.csv"))
    
if __name__ == "__main__":

    for dataset in ["amazon"]:
        for SampleTecnic in ["normal"]:
            for InvertedGraph in ["direct"]:
                args = overall_args()
                args.InvertedGraph = InvertedGraph
                args.dataset = dataset
                args.SampleTecnic = SampleTecnic
                runMethod(args)

