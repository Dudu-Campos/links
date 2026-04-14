#ifndef FUNCOES_H
#define FUNCOES_H

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

void build_bipartite(
    const std::vector<std::vector<int>>& edges,
    Eigen::SparseMatrix<bool, Eigen::ColMajor>& B,                     // saída: left x right
    std::unordered_map<int,int>& left_id_map,                         // original -> left idx
    std::unordered_map<int,int>& right_id_map,                        // original -> right idx
    std::unordered_map<int,int>& inverse_left,
    std::unordered_map<int,int>& inverse_right,
    std::unordered_map<int,int>& left_degree,
    std::unordered_map<int,int>& right_degree);


std::vector<std::vector<double>> calc_probability(
    const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
    int col1,
    int numberNeighbors,
    std::unordered_map<int,int>& inverse_left,
    std::unordered_map<int,int>& inverse_right,
    std::unordered_map<int,int>& left_degree,
    std::unordered_map<int,int>& right_degree);

void write_csv(const std::vector<std::vector<double>> vector, const std::string &Adress);

std::vector<std::vector<int>> read_csv(const std::string& Adress);

#endif