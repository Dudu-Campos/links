#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <sstream>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <string> 
#include <atomic>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <chrono>
#include <functional>
#include <limits>
#include <sys/statvfs.h>
#include "prob_calc.h"





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



inline std::vector<int> get_col_indices(
    const Eigen::SparseMatrix<int8_t, Eigen::ColMajor>& mat,
    int col
) {
    std::vector<int> indices;
    for (Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(mat, col); it; ++it)
        indices.push_back(it.row());
    return indices;
}

inline void intersect_sorted(
    const std::vector<int>& A,
    const std::vector<int>& B,
    std::vector<int>& out
) {
    out.clear();
    size_t i = 0, j = 0;

    while (i < A.size() && j < B.size()) {
        if (A[i] == B[j]) {
            out.push_back(A[i]);
            i++; j++;
        } else if (A[i] < B[j]) i++;
        else j++;
    }
}


struct VectorHasher {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for(auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};


struct ThreadCache {
    std::unordered_map<int, std::vector<int>> col_cache;




    std::unordered_map<std::vector<int>, std::vector<std::pair<int, double>>, VectorHasher> neigh_cache;
    
    const std::vector<int>& get_col(
        const Eigen::SparseMatrix<int8_t, Eigen::ColMajor>& mat,
        int col
    ) {
        auto it = col_cache.find(col);
        if (it != col_cache.end()) return it->second;

        std::vector<int> indices;
        for (Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(mat, col); it; ++it)
            indices.push_back(it.row());
        
        return col_cache.emplace(col, std::move(indices)).first->second;
    }

    void clear() {
        col_cache.clear();
        neigh_cache.clear();
    }

};


void aggregator_thread(Pipeline& pipe, 
                       std::unordered_map<EdgeKey, StatsAccumulator, EdgeKeyHash>& edge_stats,
                       std::unordered_map<int32_t, int>& col_max_size) {
    try {
        std::vector<RowProb> batch;
        while (true) {
            batch.clear();
            if (!pipe.pop_batch(batch)) break; 
            
            for (const auto& p : batch) {
                double v = p.prob;
                if (v <= 0) continue;

                EdgeKey k{p.col, p.no2}; 
                auto& st = edge_stats[k];
                st.size++;
                st.s1 += v;
                st.s2 += v * v;
                st.s3 += v * v * v;
                st.s4 += v * v * v * v;
                st.raw_entropy += -v * std::log(v);
                            
                int bin_index = static_cast<int>(p.prob * 10.0);
                if (bin_index >= 10) bin_index = 9;
                if (bin_index < 0) bin_index = 0;

                st.bins[bin_index] += 1.0f; 
                if (st.size > col_max_size[p.col]) {
                    col_max_size[p.col] = st.size;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[ERRO AGREGADOR] Falha na agregação em memória: " << e.what() << "\n";
    }
}

void build_bipartite(
    const std::vector<std::vector<int>>& edges,
    Eigen::SparseMatrix<int8_t, Eigen::ColMajor>& B,                     
    std::unordered_map<int,int>& left_id_map,                         
    std::unordered_map<int,int>& right_id_map,                        
    std::unordered_map<int,int>& inverse_left,  
    std::unordered_map<int,int>& inverse_right,  
    std::unordered_map<int,int>& degree_left,
    std::unordered_map<int,int>& degree_right,
    bool is_directed = false){

    left_id_map.clear();
    right_id_map.clear();
    inverse_right.clear();
    inverse_left.clear();

    int next_left = 0, next_right = 0;
    std::vector<Eigen::Triplet<int8_t>> triplets;

    for (const auto& row : edges) {
        if (row.size() < 2) continue;
        int u = row[0]; 
        int v = row[1];

        degree_left[u]++;
        degree_right[v]++;

        auto itl = left_id_map.find(u);
        if (itl == left_id_map.end()) { 
            left_id_map[u] = next_left; 
            inverse_left[next_left] = u; 
            ++next_left; 
        }
        
        auto itr = right_id_map.find(v);
        if (itr == right_id_map.end()) { 
            right_id_map[v] = next_right; 
            inverse_right[next_right] = v; 
            ++next_right; 
        }

        int lu = left_id_map[u];
        int rv = right_id_map[v];
        triplets.emplace_back(lu, rv, 1);
    }

    B.resize(next_left, next_right);
    B.setFromTriplets(triplets.begin(), triplets.end());
}
void calc_probability(
    const Eigen::SparseMatrix<int8_t, Eigen::ColMajor>& mat,
    int col1, 
    int fixed_idx,
    int numberNeighbors,
    const std::unordered_map<int,int>& inverse_left,  
    const std::unordered_map<int,int>& inverse_right, 
    const std::unordered_map<int,int>& left_degrees,
    const std::unordered_map<int,int>& right_degrees,
    const std::vector<double>& inv_log_right,
    const std::vector<std::vector<int>>& row_adj_list,
    const std::unordered_map<int, std::vector<int>>& target_adj_list,
    Pipeline& pipe){
        
 
    auto it_left = inverse_left.find(col1);
    if (it_left == inverse_left.end()) {
        return;
    }
    int original_col = it_left->second;

    auto target_it = target_adj_list.find(original_col);
    if (target_it == target_adj_list.end()) {
        return;
    }
    const std::vector<int>& my_targets = target_it->second; 

    auto it_deg = left_degrees.find(original_col);
    if (it_deg == left_degrees.end()) {
        std::cerr << "[ERRO] original_col=" << original_col << " não tem grau\n";
        return;
    }
    int d_left = it_deg->second;

    Eigen::SparseVector<int8_t> allNeigh = mat.row(col1);

    std::vector<int> neighList;
    for (Eigen::SparseVector<int8_t>::InnerIterator it(allNeigh); it; ++it)
       neighList.push_back(it.index());

    std::sort(neighList.begin(), neighList.end());  
    neighList.erase(std::unique(neighList.begin(), neighList.end()), neighList.end());

    size_t n = neighList.size();
    if (n < (size_t)numberNeighbors) return;

    bool use_split = (fixed_idx != -1);
    
    if (use_split && (fixed_idx >= (int)n || (n - fixed_idx - 1) < (size_t)(numberNeighbors - 1))) {
        return; 
    }

    int pool_size = use_split ? (n - fixed_idx - 1) : n;
    int k_needed  = use_split ? (numberNeighbors - 1) : numberNeighbors;

    std::vector<RowProb> local_buffer;
    local_buffer.reserve(10000);

    double eps = std::numeric_limits<double>::epsilon();
    double log_d_left = std::log(1.0 + d_left + eps);

    thread_local ThreadCache cache;
    cache.clear();

    int expected_size = mat.cols();
    if ((int)inv_log_right.size() != expected_size) return;


    std::vector<int> comb(k_needed);
    for (int i = 0; i < k_needed; ++i) comb[i] = i;

    std::vector<int> currentNeigh;
    std::vector<int> current;
    std::vector<int> next;
    std::vector<int> all_ks;
    
    currentNeigh.reserve(k_needed + 1);
    current.reserve(1000);
    next.reserve(1000);
    all_ks.reserve(5000);

    while (true) {
        currentNeigh.clear();
        current.clear();
        next.clear();
        all_ks.clear();
        
        if (use_split) currentNeigh.push_back(neighList[fixed_idx]);
        
        for (int i = 0; i < k_needed; ++i) {
            currentNeigh.push_back(neighList[(use_split ? fixed_idx + 1 : 0) + comb[i]]);
        }

        if (!currentNeigh.empty()) {
            current = cache.get_col(mat, currentNeigh[0]);
            bool valid = true;

            for (int j = 1; j < (int)currentNeigh.size(); j++) {
                const auto& col_vec = cache.get_col(mat, currentNeigh[j]);
                intersect_sorted(current, col_vec, next);
                current.swap(next);
                if (current.empty()) {
                    valid = false;
                    break;
                }
            }

            if (valid && !current.empty()) {
                for (int node : current) {
                    const auto& row_vec = row_adj_list[node];
                    all_ks.insert(all_ks.end(), row_vec.begin(), row_vec.end());
                }
                
                std::sort(all_ks.begin(), all_ks.end());

                double inv_current_size = 1.0 / (double)current.size();
                size_t idx = 0;
                
                while (idx < all_ks.size()) {
                    int k = all_ks[idx];
                    int count = 0;
                    
                    while (idx < all_ks.size() && all_ks[idx] == k) {
                        count++;
                        idx++;
                    }

                    if (k >= 0 && k < (int)inv_log_right.size()) {
                        auto it_inv_right = inverse_right.find(k);
                        if (it_inv_right != inverse_right.end()) {
                            int original_right_id = it_inv_right->second;
                            auto it_deg_right = right_degrees.find(original_right_id);
                            
                            if (!std::binary_search(my_targets.begin(), my_targets.end(), original_right_id)) {
                                continue; 
                            }
                            

                            if (it_deg_right != right_degrees.end()) {
                                int d_right = it_deg_right->second;
                                double base_scale = (d_right > 0) ? (1.0 / inv_log_right[k]) : 1.0;
                                double base_prob = (count * base_scale) * inv_current_size;
                                
                                double final_value = base_prob * log_d_left;
                                if (final_value > 1e-8) {
                                    local_buffer.push_back({original_col, original_right_id, static_cast<float>(final_value)});
                                    if (local_buffer.size() >= 10000) {
                                        pipe.push_batch(local_buffer);
                                        local_buffer.clear();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (k_needed == 0) break;
        
        
        int idx_comb = k_needed - 1;
        while (idx_comb >= 0 && comb[idx_comb] == pool_size - k_needed + idx_comb) {
            idx_comb--;
        }
        
        if (idx_comb < 0) break; 
        
        comb[idx_comb]++;
        for (int j = idx_comb + 1; j < k_needed; ++j) {
            comb[j] = comb[j - 1] + 1;
        }
    }

    if (!local_buffer.empty()) {
        pipe.push_batch(local_buffer);
        local_buffer.clear();
    }
}