#ifndef FUNCOES_H
#define FUNCOES_H

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>
#include <queue>
#include <string> 
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>

struct ResultItem {
    std::vector<double> data;
};

struct RowProb {
    int col;
    int no2;
    float prob;
};


struct EdgeKey {
    int32_t col;
    int32_t row1;
    bool operator==(const EdgeKey& other) const {
        return row1 == other.row1 && col == other.col;
    }
};


struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& k) const {
        return std::hash<int32_t>()(k.row1) ^ (std::hash<int32_t>()(k.col) << 1);
    }    
};    


struct StatsAccumulator {
    int size = 0; 
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    double s4 = 0.0;
    double raw_entropy = 0.0;
    int bins[10] = {0}; // NOVO: Guarda a contagem em 10 intervalos
};

#pragma pack(push, 1)
struct InputEdge {
    int32_t col;
    int32_t row1;
    float prob;
};

struct OutputEdge {
    int32_t col;
    int32_t row1;
    float kurt_skew;
    float entropy;
};
#pragma pack(pop)



struct Pipeline {
    std::queue<RowProb> queue;
    std::mutex mtx;
    std::condition_variable cv_consumer; // Notifica o consumidor
    std::condition_variable cv_producer; // Notifica os produtores
    bool finished = false;
    const size_t max_queue_size = 500000; // Limite de segurança (ajuste conforme sua RAM)

    void push_batch(const std::vector<RowProb>& local_buffer) {
        if (local_buffer.empty()) return; // IMPRESCINDÍVEL: Não travar por buffers vazios
        
        std::unique_lock<std::mutex> lock(mtx);
        // Se já terminou, não vale a pena esperar por espaço
        if (finished) return;

        cv_producer.wait(lock, [this, &local_buffer] {
            return (queue.size() + local_buffer.size() <= max_queue_size) || finished;
        });

        if (finished) return; 

        for (const auto& item : local_buffer) {
            queue.push(item);
        }
        cv_consumer.notify_one(); 
    }

    // Atualize o push individual também se ainda o utilizar
    void push(int colInic, int row, float prob) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_producer.wait(lock, [this] { return queue.size() < max_queue_size || finished; });
        
        queue.push({colInic, row, prob});
        cv_consumer.notify_one();
    }

    bool pop_batch(std::vector<RowProb>& buffer) {
        std::unique_lock<std::mutex> lock(mtx);

        cv_consumer.wait(lock, [this] {
            return !queue.empty() || finished;
        });

        if (queue.empty() && finished)
            return false;

        while (!queue.empty()) {
            buffer.push_back(queue.front());
            queue.pop();
        }

        // Notifica produtores que agora há espaço na fila
        cv_producer.notify_all(); 
        return true;
    }
};

struct Prediction {
    int u;
    int v;
    float prob;
};



void aggregator_thread(Pipeline& pipe, 
                       std::unordered_map<EdgeKey, StatsAccumulator, EdgeKeyHash>& edge_stats,
                       std::unordered_map<int32_t, int>& col_max_size);
  


void build_homogeneous(
    const std::vector<std::vector<int>>& edges,
    Eigen::SparseMatrix<bool, Eigen::ColMajor>& B,      // saída: matriz N x N
    std::unordered_map<int,int>& node_id_map,           // original -> idx
    std::unordered_map<int,int>& inverse_map,           // idx -> original
    std::unordered_map<int,int>& degree_map,            // original -> grau total
    bool is_directed);     


void build_bipartite(
    const std::vector<std::vector<int>>& edges,
    Eigen::SparseMatrix<bool, Eigen::ColMajor>& B,                     
    std::unordered_map<int,int>& left_id_map,                         
    std::unordered_map<int,int>& right_id_map,                        
    std::unordered_map<int,int>& inverse_left,   
    std::unordered_map<int,int>& inverse_right, 
    std::unordered_map<int,int>& degree_left,
    std::unordered_map<int,int>& degree_right,
    bool is_directed);

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
    Pipeline& pipe);


void write_csv(const std::vector<std::vector<double>> vector, const std::string &Adress);

std::vector<std::vector<int>> read_csv(const std::string& Adress);

#endif
