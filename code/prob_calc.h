#ifndef FUNCOES_H
#define FUNCOES_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include <Eigen/Sparse>
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
    int node;
    double prob;
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
    double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    double raw_entropy = 0;
    int size = 0;
    float bins[10] = {0};
};


struct InputEdge {
    int32_t col;
    int32_t row1;
    float prob;
};



struct OutputEdge {
    int32_t col;
    int32_t row1;
    float kurt;
    float skew;
    float entropy;
};
struct Pipeline {
    std::queue<RowProb> queue;
    std::mutex mtx;
    std::condition_variable cv_consumer; 
    std::condition_variable cv_producer; 
    bool finished = false;
    const size_t max_queue_size = 10000;

    void push_batch(const std::vector<RowProb>& local_buffer) {
        if (local_buffer.empty()) return; 
        
        std::unique_lock<std::mutex> lock(mtx);
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

        cv_producer.notify_all(); 
        return true;
    }
};

struct Prediction {
    int u;
    int v;
    float prob;
};


struct TargetHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

void aggregator_thread(Pipeline& pipe, 
                       std::unordered_map<EdgeKey, StatsAccumulator, EdgeKeyHash>& edge_stats,
                       std::unordered_map<int32_t, int>& col_max_size);
  


void build_homogeneous(
    const std::vector<std::vector<int>>& edges,
    Eigen::SparseMatrix<int8_t, Eigen::ColMajor>& B,     
    std::unordered_map<int,int>& node_id_map,          
    std::unordered_map<int,int>& inverse_map,           
    std::unordered_map<int,int>& degree_map,           
    bool is_directed);     


void build_bipartite(
    const std::vector<std::vector<int>>& edges,
    Eigen::SparseMatrix<int8_t, Eigen::ColMajor>& B,                     
    std::unordered_map<int,int>& left_id_map,                         
    std::unordered_map<int,int>& right_id_map,                        
    std::unordered_map<int,int>& inverse_left,   
    std::unordered_map<int,int>& inverse_right, 
    std::unordered_map<int,int>& degree_left,
    std::unordered_map<int,int>& degree_right,
    bool is_directed);

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
    Pipeline& pipe);


void write_csv(const std::vector<std::vector<double>> vector, const std::string &Adress);

std::vector<std::vector<int>> read_csv(const std::string& Adress);

#endif
