# Conditional Space Graph Representation

**Conference:** BRACIS2026
---

## Description

New approach implementation to graph representation. Feature space that encodes co-occurrences links.

## Repository

```text
├── data/        
├────citeulike              # Dataset
├────lastfm                 # Dataset
├────ml-100k                # Dataset
├────exp                    # Experiments results
├────kpisti-L3-a163e9f      # Provided L3 implementation 
├── code/                   # implementation
├────args.py                # runs parameters
├────create_graph.py        # graph train/test split
├────Experiments.py         # run evaluation
├────OurMethodExp.cpp       # calculate CSGR features
├────prob_calc.cpp          # calculate CSGR features
├────prob_calc.h            # calculate CSGR features
└── requirements.txt

### Executing program

```
python3 "code/Experiments.py"
```

### Args description

the default parameters are the ones used in the article

 *exec_path : where files are salved, changing this parameter will break the code.
 *d :  used to choose a single dataset.
 *F : minimum degree Filter, not used in the inicial work.
 *perct : train/test split ratio.
 *Ns : NeighSize parameter described in the paper
 *Pr : PruningRatio only calculate features to most similar nodes, not used in the inicial work.
 *St : used to choose a single sample to create the dataset.
 *Iv : invert graph representation, not used in the inicial work.
 *Rs : list of seeds to run the evaluation
 *Cs : current seed


## License

This project is licensed under the **MIT License**. 

*Copyright (c) 2026 The Authors (Anonymized for Peer Review)*

See the full license text below:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.