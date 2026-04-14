from pathlib import Path
from argparse import ArgumentParser


column_name_update = {
    "movies/rating": ["userId","movieId"],
    "lastfm/user_artists": ["userID","artistID"]


}

dataset_format = {
    "movies/rating" : [",","csv"],
    "lastfm/user_artists" : ["\t","dat"]
}



def overall_args():
    parser = ArgumentParser()

    parser.add_argument("--exec_path", dest="exec_path", type=str,nargs='?', default=f'{Path.cwd()}')
    parser.add_argument("--Gs",dest="GraphSize",type=int, default=1450)
    parser.add_argument("--d",dest="dataset",type=str,default="ml-100k")
    parser.add_argument("--perct", dest = "perct",type=float, default = 0.1)
    parser.add_argument("--Ns", dest = "NeighSize",type=int, default = 1)
    parser.add_argument("--Pr", dest = "PruningRatio",type=int, default = 1)
    parser.add_argument("--St", dest = "SampleTecnic",type=str, default = "normal")
    parser.add_argument("--Iv", dest = "InvertedGraph",type=str, default = "direct")
    parser.add_argument("--Rs", dest = "RandomSeed",type=int,default= 36)



    args = parser.parse_args()
    return args

