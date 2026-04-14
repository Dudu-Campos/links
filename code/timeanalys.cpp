#include <iostream>
#include <random>
#include <cstdlib>
#include <string>
#include <chrono>
#include <ctime>
#include "prob_calc.h"

int main() {
    double argv[5] = {0,100,200,0.05,42};

    for(int it=1;it<6;it++){
        int nU = pow(2,10);
        int nV = pow(2,10);
        std::vector<std::vector<int>> edges;
        std::unordered_map<int,int> degreeCounter;
        double p;
        p = argv[3];

        unsigned seed;
        seed = static_cast<unsigned>(argv[4]);

        std::mt19937_64 rng(seed);
        std::bernoulli_distribution edge_prob(p);

        for (int i = 0; i < nU; ++i) {
            degreeCounter[i] = 0;
        }
        
        for (int i = 0; i < nU; ++i) {
            for (int j = 0; j < nV; ++j) {
                if (edge_prob(rng)) { 
                    degreeCounter[i]++;
                    edges.push_back({i,j}) ;
                }
            }
        }

        Eigen::SparseMatrix<bool, Eigen::ColMajor> B;
        std::unordered_map<int,int> left_id_map, right_id_map, inverse_left, inverse_right;
        build_bipartite(edges,B,left_id_map, right_id_map, inverse_left, inverse_right);
        auto start = std::chrono::steady_clock::now();
        calc_probability2(B,0,it,inverse_left,inverse_right);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << it << "," << duration.count() << std::endl;
    }

    return 0;
}