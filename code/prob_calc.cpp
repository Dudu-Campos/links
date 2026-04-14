#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>
#include <string> 
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include "prob_calc.h"
#include <cmath>
#include <limits>

 std::vector<std::vector<int>> read_csv(const std::string& Adress){
    
    std::string line;
    std::vector<std::vector<int>> values;
    std::ifstream file(Adress); 

    while (std::getline(file, line)) {
            std::stringstream ss(line); 
            std::string cell;
            std::vector<int> row;

            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stoi(cell)); 
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number format in line: " << line << std::endl;
                }
            }
            values.push_back(row);
        }

        file.close();

            return values;
        }

void write_csv(std::vector<std::vector<double>> vec, const std::string& Adress){

    std::ofstream MyFile(Adress);
    if(!MyFile.is_open()){
        std::cout<< "erro" << std::endl;
        MyFile << "11111";
    }
    for (const auto& row : vec){
        for(size_t i=0; i<row.size(); ++i){
            MyFile << row[i];
            if (i + 1 < row.size()) MyFile << ",";
        }
        MyFile << std::endl;
    }
    MyFile.close();

}

Eigen::SparseVector<bool> intersecao_colunas_otimizada(
    const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
    int col1,
    const Eigen::SparseVector<bool>& col2)
{
    Eigen::SparseVector<bool> resultado(mat.rows());
    resultado.reserve(std::min(mat.col(col1).nonZeros(),
                               col2.nonZeros()));

    Eigen::SparseMatrix<bool, Eigen::ColMajor>::InnerIterator it1(mat, col1);
    Eigen::SparseVector<bool>::InnerIterator it2(col2);

    // iterate both sorted lists and collect common row indices
    while (it1 && it2) {
        int r1 = it1.row();
        int r2 = it2.index();
        if (r1 == r2) {
            resultado.coeffRef(r1) = true;
            ++it1;
            ++it2;
        }
        else if (r1 < r2) {
            ++it1;
        }
        else {
            ++it2;
        }
    }

    return resultado;
}


Eigen::SparseVector<bool> intersecao_linha_otimizada(
    const Eigen::SparseMatrix<bool, Eigen::RowMajor>& mat,
    int row1,
    const Eigen::SparseVector<bool>& row2)
{
    Eigen::SparseVector<bool> resultado(mat.cols());
    resultado.reserve(std::min(mat.row(row1).nonZeros(),
                               row2.nonZeros()));

    Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it1(mat, row1);
    Eigen::SparseVector<bool>::InnerIterator it2(row2);

    // iterate both sorted lists and collect common row indices
    while (it1 && it2) {
        int r1 = it1.row();
        int r2 = it2.index();
        if (r1 == r2) {
            resultado.coeffRef(r1) = true;
            ++it1;
            ++it2;
        }
        else if (r1 < r2) {
            ++it1;
        }
        else {
            ++it2;
        }
    }

    return resultado;
}



Eigen::SparseVector<bool> intersecao_col_otimizada(
    const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
    int col1,
    const Eigen::SparseVector<bool>& col2)
{
    Eigen::SparseVector<bool> resultado(mat.rows());
    resultado.reserve(std::min(mat.col(col1).nonZeros(),
                               col2.nonZeros()));

    Eigen::SparseMatrix<bool, Eigen::ColMajor>::InnerIterator it1(mat, col1);
    Eigen::SparseVector<bool>::InnerIterator it2(col2);

    // iterate both sorted lists and collect common row indices
    while (it1 && it2) {
        int r1 = it1.col();
        int r2 = it2.index();
        if (r1 == r2) {
            resultado.coeffRef(r1) = true;
            ++it1;
            ++it2;
        }
        else if (r1 < r2) {
            ++it1;
        }
        else {
            ++it2;
        }
    }

    return resultado;
}


std::vector<std::vector<double>> calc_probability(
    const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
    int col1,
    int numberNeighbors,
    std::unordered_map<int,int>& inverse_right,
    std::unordered_map<int,int>& inverse_left,
    std::unordered_map<int,int>& left_degrees,
    std::unordered_map<int,int>& right_degrees)
{
    // all neighbors of the left node 'col1' (row index)
    Eigen::SparseVector<bool> allNeigh = mat.row(col1);
     std::unordered_map<int,int> prunningSet;
    std::unordered_map<int,int> zeroCounter;
     std::vector<int> neighList;
     int iterações =0;



    for (Eigen::SparseVector<bool>::InnerIterator it(allNeigh); it; ++it) {
         
         neighList.push_back(it.row());
         
         
     }
     for(int i=0;i<mat.cols();i++){
         prunningSet[inverse_left[i]]=1;
         zeroCounter[i]=0;
     }
     
     std::vector<std::vector<double>> results;
     
     
     size_t n = neighList.size();
     std::vector<int> sel(n, 0);
     for (size_t i = n - numberNeighbors; i < n; ++i) sel[i] = 1;
     
     
     do {
         std::vector<int> currentNeigh;
         currentNeigh.reserve(numberNeighbors);
         iterações++;
         for (size_t idx = 0; idx < n; ++idx) {
             if (sel[idx]) currentNeigh.push_back(neighList[idx]);
         }
         if (currentNeigh.empty()) continue;
         
         
         Eigen::SparseVector<bool> UnionVec = mat.col(currentNeigh[0]);
 
         if(numberNeighbors >1){
             for(int j=1;j<currentNeigh.size();j++){
                 UnionVec = intersecao_colunas_otimizada(mat,currentNeigh[j],UnionVec);
             }
         }
         
         
         std::vector<int> nodes_list;
         
         for (Eigen::SparseVector<bool>::InnerIterator it(UnionVec); it; ++it) {
                 nodes_list.push_back(it.row());
             } 
             std::vector<int> probs(mat.cols(),0);
             if(nodes_list.size()>0){
                 
                 for(int node=0;node<nodes_list.size();node++){
                     
                     Eigen::SparseVector<bool> valuesColumn = mat.row(nodes_list[node]);
                     for (Eigen::SparseVector<bool>::InnerIterator it(valuesColumn); it; ++it) {
                         probs[it.row()]++;
 
                     } 
 
 
                 }

                // normalize and scale probs by degree ratio: left node degree vs candidate right node degree
                double eps = std::numeric_limits<double>::epsilon();
                int d_left = left_degrees[inverse_right[col1]]; // degree of the source left node
                for(int k=0;k<static_cast<int>(probs.size());k++){
                    if(probs[k] == 0){
                        zeroCounter[inverse_left[k]]++;
                    }

                    // only include if not pruned
                    if(prunningSet[inverse_left[k]]){
                        int d_right = right_degrees[inverse_left[k]];
                        double scale = 1.0;
                        if (d_right > 0) scale = std::log(1.0 + static_cast<double>(d_left) + eps) / std::log(1.0 + static_cast<double>(d_right) + eps);
                        double value = (static_cast<double>(probs[k]) * scale) / static_cast<double>(nodes_list.size());
                        results.push_back({ static_cast<double>(inverse_right[col1]),
                                            static_cast<double>(inverse_left[k]),
                                            value });
                    }
                }
             }
             else{
                 for(int k=0;k<probs.size();k++){
                     zeroCounter[inverse_left[k]]++;
                     results.push_back({static_cast<double>(inverse_right[col1]),static_cast<double>(inverse_left[k]),0});
                     }
                 }
 
             
 
            
     } while (std::next_permutation(sel.begin(), sel.end()));
     
     std::vector<std::vector<double>> filter_results;
     for(int i =0;i<results.size();i++){
         if(prunningSet[results[i][1]] == 1){
             filter_results.push_back(results[i]);
         }
     }
 
 
 
 
             return filter_results;
          }
 

void build_bipartite(
    const std::vector<std::vector<int>>& edges,
    Eigen::SparseMatrix<bool, Eigen::ColMajor>& B,                     // saída: left x right
    std::unordered_map<int,int>& left_id_map,                         // original -> left idx
    std::unordered_map<int,int>& right_id_map,                        // original -> right idx
    std::unordered_map<int,int>& inverse_right,
    std::unordered_map<int,int>& inverse_left,
    std::unordered_map<int,int>& degree_left,
    std::unordered_map<int,int>& degree_right)
{
    left_id_map.clear();
    right_id_map.clear();
    inverse_right.clear();
    inverse_left.clear();

    int next_left = 0, next_right = 0;
    std::vector<Eigen::Triplet<bool>> triplets;

    for (const auto& row : edges) {
        if (row.size() < 2) continue;
        int u = row[0]; // assume u in left
        int v = row[1]; // assume v in right

        degree_left[u]++;
        degree_right[v]++;

        auto itl = left_id_map.find(u);
        if (itl == left_id_map.end()) { left_id_map[u] = next_left; inverse_right[next_left] = u; ++next_left; }
        auto itr = right_id_map.find(v);
        if (itr == right_id_map.end()) { right_id_map[v] = next_right; inverse_left[next_right] = v; ++next_right; }

        int lu = left_id_map[u];
        int rv = right_id_map[v];
        triplets.emplace_back(lu, rv, true);
    }

    B.resize(next_left, next_right);
    B.setFromTriplets(triplets.begin(), triplets.end());


}

