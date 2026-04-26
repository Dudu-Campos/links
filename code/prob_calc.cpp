#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <fstream>
#include <sstream>
#include <set>
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
#include <eigen3/Eigen/Sparse>
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
    const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
    int col
) {
    std::vector<int> indices;
    for (Eigen::SparseMatrix<bool, Eigen::ColMajor>::InnerIterator it(mat, col); it; ++it)
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
    std::unordered_map<int, std::vector<int>> row_cache;




    std::unordered_map<std::vector<int>, std::vector<std::pair<int, double>>, VectorHasher> neigh_cache;
    
    const std::vector<int>& get_col(
        const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
        int col
    ) {
        auto it = col_cache.find(col);
        if (it != col_cache.end()) return it->second;

        std::vector<int> indices;
        for (Eigen::SparseMatrix<bool, Eigen::ColMajor>::InnerIterator it(mat, col); it; ++it)
            indices.push_back(it.row());
        
        return col_cache.emplace(col, std::move(indices)).first->second;
    }

    const std::vector<int>& get_row(
        const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
        int row
    ) {
        auto it = row_cache.find(row);
        if (it != row_cache.end()) return it->second;

        std::vector<int> indices;
        // Puxa a linha específica e itera sobre as colunas (nós direitos) conectados
        Eigen::SparseVector<bool> row_vec = mat.row(row);
        for (Eigen::SparseVector<bool>::InnerIterator iter(row_vec); iter; ++iter)
            indices.push_back(iter.index());

        return row_cache.emplace(row, std::move(indices)).first->second;
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
                st.raw_entropy += -v * std::log(v); // Nota: Use std::log em vez de std::log2 para evitar aquele erro de domínio matemático antigo
                            
                // === Lógica do Histograma CORRIGIDA ===
                int bin_index = static_cast<int>(p.prob * 10.0);
                if (bin_index >= 10) bin_index = 9;
                if (bin_index < 0) bin_index = 0;

                // Incrementa APENAS a casa (bin) onde esta probabilidade caiu
                st.bins[bin_index] += 1.0f; // Adiciona +1 à contagem desta faixa

                // Atualiza o tamanho máximo
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
    Eigen::SparseMatrix<bool, Eigen::ColMajor>& B,                     
    std::unordered_map<int,int>& left_id_map,                         
    std::unordered_map<int,int>& right_id_map,                        
    std::unordered_map<int,int>& inverse_left,   // <-- ESQUERDA AQUI
    std::unordered_map<int,int>& inverse_right,  // <-- DIREITA AQUI
    std::unordered_map<int,int>& degree_left,
    std::unordered_map<int,int>& degree_right,
    bool is_directed = false)
{

    left_id_map.clear();
    right_id_map.clear();
    inverse_right.clear();
    inverse_left.clear();

    int next_left = 0, next_right = 0;
    std::vector<Eigen::Triplet<bool>> triplets;

    for (const auto& row : edges) {
        if (row.size() < 2) continue;
        int u = row[0]; // assume u in left (Usuário)
        int v = row[1]; // assume v in right (Item)

        degree_left[u]++;
        degree_right[v]++;

        // CORRIGIDO: Esquerda vai para inverse_left
        auto itl = left_id_map.find(u);
        if (itl == left_id_map.end()) { 
            left_id_map[u] = next_left; 
            inverse_left[next_left] = u; 
            ++next_left; 
        }
        
        // CORRIGIDO: Direita vai para inverse_right
        auto itr = right_id_map.find(v);
        if (itr == right_id_map.end()) { 
            right_id_map[v] = next_right; 
            inverse_right[next_right] = v; 
            ++next_right; 
        }

        int lu = left_id_map[u];
        int rv = right_id_map[v];
        triplets.emplace_back(lu, rv, true);
    }

    B.resize(next_left, next_right);
    B.setFromTriplets(triplets.begin(), triplets.end());
}

void calc_probability(
    const Eigen::SparseMatrix<bool, Eigen::ColMajor>& mat,
    int col1, 
    int fixed_idx,
    int numberNeighbors,
    std::unordered_map<int,int>& inverse_left,  
    std::unordered_map<int,int>& inverse_right, 
    std::unordered_map<int,int>& left_degrees,
    std::unordered_map<int,int>& right_degrees,
    const std::vector<double>& inv_log_right,
    Pipeline& pipe){

    Eigen::SparseVector<bool> allNeigh = mat.row(col1);

    std::vector<int> neighList;
    for (Eigen::SparseVector<bool>::InnerIterator it(allNeigh); it; ++it)
        neighList.push_back(it.row());

    size_t n = neighList.size();
    if (n < (size_t)numberNeighbors) return;

    std::sort(neighList.begin(), neighList.end());
    neighList.erase(std::unique(neighList.begin(), neighList.end()), neighList.end());


    // ==========================================================
    // LÓGICA DE FATIAMENTO DE COMBINAÇÕES
    // ==========================================================
    bool use_split = (fixed_idx != -1);
    
    // Proteções: Se a estimativa do grau falhou ou não sobraram vizinhos para formar K itens
    if (use_split && (fixed_idx >= n || (n - fixed_idx - 1) < (size_t)(numberNeighbors - 1))) {
        return; 
    }

    // Se estivermos a dividir, só precisamos de (NeighSize - 1) do que sobrou à direita!
    int pool_size = use_split ? (n - fixed_idx - 1) : n;
    int k_needed  = use_split ? (numberNeighbors - 1) : numberNeighbors;

    std::vector<int> sel(pool_size, 0);
    for (int i = pool_size - k_needed; i < pool_size; ++i) sel[i] = 1;


    std::vector<RowProb> local_buffer;
    local_buffer.reserve(10000);


    std::sort(neighList.begin(), neighList.end());
    neighList.erase(std::unique(neighList.begin(), neighList.end()), neighList.end());

  

    double eps = std::numeric_limits<double>::epsilon();
    
    // CORRIGIDO: Agora usamos inverse_left para pegar o nó da esquerda real
    int original_col = inverse_left[col1]; 
    int d_left = left_degrees[original_col];
    
    // 1. Pré-calculamos o log do nó da esquerda UMA VEZ
    double log_d_left = std::log(1.0 + d_left + eps);

    static thread_local ThreadCache cache;

    do {
        std::vector<int> currentNeigh;
        
        if (use_split) {
            // Para as fatias: O primeiro elemento é fixo!
            currentNeigh.push_back(neighList[fixed_idx]);
            for (int i = 0; i < pool_size; i++) {
                if (sel[i]) currentNeigh.push_back(neighList[fixed_idx + 1 + i]); // Pega do resto
            }
        } else {
            // Lógica antiga para nós pequenos
            for (int i = 0; i < pool_size; i++) {
                if (sel[i]) currentNeigh.push_back(neighList[i]);
            }
        }

        if (currentNeigh.empty()) continue;

        auto cache_it = cache.neigh_cache.find(currentNeigh);
        
        if (cache_it != cache.neigh_cache.end()) {
            for (const auto& par : cache_it->second) {
                int k = par.first; // k é o nó da direita interno
                double base_prob = par.second;
                
                double final_value = base_prob*log_d_left;
                
          if (final_value > 1e-8) {
            local_buffer.push_back({original_col, inverse_right[k], static_cast<float>(final_value)});
            if (local_buffer.size() >= 10000) {
                pipe.push_batch(local_buffer); // Isso agora bloqueará se a fila global estiver cheia
                local_buffer.clear(); 
            }
        }
    }
            continue; 
        }
            
        

        std::vector<int> current = cache.get_col(mat, currentNeigh[0]);
        std::vector<int> next;

        for (int j = 1; j < (int)currentNeigh.size(); j++) {
            const auto& col_vec = cache.get_col(mat, currentNeigh[j]);
            intersect_sorted(current, col_vec, next);
            current.swap(next);
            if (current.empty()) break;
        }

        if (current.empty()) continue;

        std::unordered_map<int,int> probs;
        for (int node : current) {
            const auto& row_vec = cache.get_row(mat, node);
            for (int k : row_vec) {
                probs[k]++;
            }
        }

        std::vector<std::pair<int, double>> resultados_base;
        double inv_current_size = 1.0 / (double)current.size();

        for (auto& [k, count] : probs) {
            double base_scale = 1.0;
            
            // CORRIGIDO: Usamos inverse_right para achar o grau do item K
            int d_right = right_degrees[inverse_right[k]];

            if (d_right > 0) {
               try {
                    // Tabela está 100% alinhada com k
                    base_scale = 1/inv_log_right.at(k);
                } catch (const std::out_of_range& e) {
                    std::cerr << "ERRO FATAL: k = " << k << " está fora do vetor de tamanho " << inv_log_right.size() << "\n";
                    exit(1);
                }
            }

            double base_prob = (count * base_scale) * inv_current_size;
            resultados_base.emplace_back(k, base_prob);
            // double final_value = base_prob;
            double final_value = base_prob*log_d_left;


            if (final_value > 1e-8) {
                // CORRIGIDO: Envia o original_col e inverse_right[k]
                pipe.push(original_col, inverse_right[k], static_cast<float>(final_value));
            }
        }

        cache.neigh_cache.emplace(currentNeigh, std::move(resultados_base));

        if (cache.neigh_cache.size() > 10000) {
        cache.neigh_cache.clear();
    }

    }  while (std::next_permutation(sel.begin(), sel.end()));

    if (!local_buffer.empty()) {
        pipe.push_batch(local_buffer);
    }
}

