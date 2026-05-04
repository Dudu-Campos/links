#include <iostream>
#include <random>
#include <cstdlib>
#include <string>
#include <chrono>
#include <ctime>
#include <iterator> 
#include <cassert>
#include <unordered_set>
#include "prob_calc.h"
#include <filesystem>
#include <sys/statvfs.h> 
#include <future>
#include <mutex>
#include <queue>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <set>
#include <unordered_map>
#include <thread>
#include <functional>
#include <cmath>
#include <limits>
#include <condition_variable>
#include <atomic>

struct OutputEdgeHist {
    int32_t col;
    int32_t row1;
    float bins[10]; 
};

struct TaskDef {
    int col;
    int fixed_idx;
    int peso; 
};

bool has_enough_disk_space(const std::string& filepath, double min_free_fraction = 0.10) {
    std::string dir = filepath;
    auto pos = dir.rfind('/');
    if (pos == std::string::npos) dir = ".";
    else if (pos == 0) dir = "/";
    else dir = dir.substr(0, pos);

    struct statvfs st;
    if (statvfs(dir.c_str(), &st) != 0) return false;

    unsigned long long total = (unsigned long long)st.f_blocks * st.f_frsize;
    unsigned long long avail = (unsigned long long)st.f_bavail * st.f_frsize;

    if (total == 0) return false;
    return ((double)avail / (double)total) > min_free_fraction;
}

void writer_thread(Pipeline& pipe, std::string filename) {
    std::ofstream out(filename, std::ios::binary);
    std::vector<RowProb> batch;
    while (true) {
        batch.clear();
        if (!pipe.pop_batch(batch)) break; 
        
        for (const auto& p : batch) {
            out.write((char*)&p.col, sizeof(int));
            out.write((char*)&p.no2, sizeof(int));
            out.write((char*)&p.prob, sizeof(float));
        }
    }
    out.close();
}

int main(int argc, char* argv[]){
    if (argc < 6) {
        std::cerr << "Uso: " << argv[0] << " <TrainAdress> <SaveAdress> <NeighSize> <Seed> < >\n";
        std::cerr << "Modo: 0 = Kurt/Skew | 1 = Writer Bruto | 2 = Histogramas (10 Bins)\n";
        return 1;
    }

    std::string TrainAdress = argv[1];
    std::string SaveAdress = argv[2];
    int NeighSize = std::stoi(argv[3]);
    std::string seed = argv[4];
    int mode_flag = std::stoi(argv[5]);
    std::string TargetAdress = argv[6];
    bool use_writer = 0;

    std::unordered_map<int, std::vector<int>> target_adj_list;
    std::ifstream t_file(TargetAdress);
    std::string t_line;
    
    std::cout << "Carregando alvos (targets) agrupados...\n";
    while (std::getline(t_file, t_line)) {
        std::stringstream t_ss(t_line);
        std::string u_str, v_str;
        if (std::getline(t_ss, u_str, ',') && std::getline(t_ss, v_str, ',')) {
            try {
                target_adj_list[std::stoi(u_str)].push_back(std::stoi(v_str));
            } catch (...) {}
        }
    }
    
    int total_targets = 0;
    for (auto& pair : target_adj_list) {
        std::sort(pair.second.begin(), pair.second.end());
        total_targets += pair.second.size();
    }
    std::cout << "Foram carregados " << total_targets << " pares alvo otimizados.\n";

    std::filesystem::create_directories(SaveAdress);

    Eigen::SparseMatrix<int8_t, Eigen::ColMajor> B;
    std::unordered_map<int,int> left_id_map, right_id_map, inverse_left, inverse_right;
    std::unordered_map<int,int> left_degree, right_degree;
    
    auto data = read_csv(TrainAdress);
    bool is_directed = false;
    build_bipartite(data, B, left_id_map, right_id_map, inverse_left, inverse_right, left_degree, right_degree, is_directed);


    std::vector<double> inv_log_right(inverse_right.size(), 1.0);
    double eps = std::numeric_limits<double>::epsilon();

    for (int k = 0; k < (int)inverse_right.size(); ++k) {
        int original_v = inverse_right[k]; 
        int d_right = right_degree[original_v]; 
        
        if (d_right > 0) {
            inv_log_right[k] =  std::log(1.0 + d_right + eps);
        }
    }   

    std::vector<std::vector<int>> row_adj_list(inverse_right.size() + inverse_left.size());

    for (int k = 0; k < B.outerSize(); ++k) {
        for (Eigen::SparseMatrix<int8_t>::InnerIterator it(B, k); it; ++it) {
            row_adj_list[it.row()].push_back(it.col()); 
        }
    }

    std::vector<TaskDef> tasks;
    std::atomic<size_t> current_task{0};
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 12;

    std::vector<std::thread> workers;
    for (int col = 0; col < left_id_map.size() ; ++col) {
        int deg = left_degree[inverse_left[col]];
        tasks.push_back({col, -1, deg}); 
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::unordered_map<EdgeKey, StatsAccumulator, EdgeKeyHash>> all_edge_stats(num_threads);
    std::vector<std::unordered_map<int32_t, int>> all_col_max_size(num_threads);

    std::sort(tasks.begin(), tasks.end(), [](const TaskDef& a, const TaskDef& b) {
        return a.peso > b.peso; 
    });

    for (unsigned int i = 0; i < num_threads; ++i) {
        workers.emplace_back([&, i]() {
            Pipeline local_pipe;
            std::thread consumer_thread;

            if (use_writer) {
                std::string out_name = SaveAdress + "/" + std::to_string(NeighSize) +"_raw_probs_thread_" + std::to_string(i) + ".bin";
                consumer_thread = std::thread(writer_thread, std::ref(local_pipe), out_name);
            } else {
                consumer_thread = std::thread(aggregator_thread, std::ref(local_pipe), 
                                              std::ref(all_edge_stats[i]), 
                                              std::ref(all_col_max_size[i]));
            }
            
            try {
                while (true) {
                    size_t idx = current_task.fetch_add(1);
                    if (idx >= tasks.size()) break; 

                    TaskDef task = tasks[idx];
                    calc_probability(
                        B, task.col, task.fixed_idx, NeighSize,
                        inverse_left, inverse_right,
                        left_degree, right_degree,
                        inv_log_right,row_adj_list,target_adj_list,local_pipe
                    );
                }
            } catch (const std::exception& e) {
                std::cerr << "\n[ERRO] Worker " << i << " falhou com exceção: " << e.what() << "\n";
            } catch (...) {
                std::cerr << "\n[ERRO] Worker " << i << " sofreu uma falha desconhecida!\n";
            }

            {
                std::lock_guard<std::mutex> lock(local_pipe.mtx);
                local_pipe.finished = true;
            }
            
            local_pipe.cv_consumer.notify_all(); 
            local_pipe.cv_producer.notify_all();
            
            if (consumer_thread.joinable()) {
                consumer_thread.join();
            }
        });
    }

    for (auto &t : workers) {
        if (t.joinable()) t.join();
    }
    

    if (!use_writer) {
        std::cout << "Agregação local concluída. Fundindo mapas paralelamente...\n";

        std::vector<std::unordered_map<EdgeKey, StatsAccumulator, EdgeKeyHash>> global_partitions(num_threads);
        std::vector<std::unordered_map<int32_t, int>> global_col_max_partitions(num_threads);

        size_t total_estimado = 0;
        for (unsigned int i = 0; i < num_threads; ++i) {
            total_estimado += all_edge_stats[i].size();
        }
        
        for (unsigned int i = 0; i < num_threads; ++i) {
            global_partitions[i].reserve((total_estimado / num_threads) + 10000);
        }

        std::vector<std::thread> merge_workers;
        for (unsigned int p = 0; p < num_threads; ++p) {
            merge_workers.emplace_back([&, p]() {
                for (unsigned int i = 0; i < num_threads; ++i) {
                    for (const auto& [key, st] : all_edge_stats[i]) {
                        if ((key.col % num_threads) == p) {
                            auto& global_st = global_partitions[p][key];
                            global_st.size += st.size;
                            global_st.s1 += st.s1;
                            global_st.s2 += st.s2;
                            global_st.s3 += st.s3;
                            global_st.s4 += st.s4;
                            global_st.raw_entropy += st.raw_entropy;

                        
                        
                            
                            float prob = static_cast<float>(st.s1 / (st.size > 0 ? st.size : 1.0));
                            int bin_index = 0;

                            if (prob >= 0.85f) {
                                bin_index = 9;    
                            } else if (prob >= 0.70f) {
                                bin_index = 8;    
                            } else if (prob >= 0.55f) {
                                bin_index = 7;    
                            } else if (prob >= 0.40f) {
                                bin_index = 6;    
                            } else if (prob >= 0.30f) {
                                bin_index = 5;     
                            } else if (prob >= 0.20f) {
                                bin_index = 4;      
                            } else if (prob >= 0.10f) {
                                bin_index = 3;     
                            } else if (prob >= 0.05f) {
                                bin_index = 2;      
                            } else if (prob >= 0.01f) {
                                bin_index = 1;     
                            } else {
                                bin_index = 0;      
                            }
                            global_st.bins[bin_index] += 1.0f;
                        
                    }






                    }
                }
                
                for (const auto& [key, st] : global_partitions[p]) {
                    if (st.size > global_col_max_partitions[p][key.col]) {
                        global_col_max_partitions[p][key.col] = st.size;
                    }
                }
            });
        }
        for (auto& t : merge_workers) { if (t.joinable()) t.join(); }

        std::cout << "Calculando Features paralelamente e gravando arquivo...\n";

        std::string out_filename = (mode_flag == 2) ? "_final_histograms_output.bin" : "_final_kurtosis_output.bin";
        std::ofstream out_file(SaveAdress + "/" + std::to_string(NeighSize) + out_filename, std::ios::binary);
        std::mutex file_mtx;
        
        std::vector<std::thread> math_workers;
        for (unsigned int p = 0; p < num_threads; ++p) {
            math_workers.emplace_back([&, p]() {
                
                std::vector<OutputEdge> local_buffer_kurt;
                std::vector<OutputEdgeHist> local_buffer_hist;

                if (mode_flag == 0) local_buffer_kurt.reserve(global_partitions[p].size());
                if (mode_flag == 2) local_buffer_hist.reserve(global_partitions[p].size());

                for (const auto& [key, st] : global_partitions[p]) {
                    
                    if (mode_flag == 2) {
                        OutputEdgeHist out_hist;
                        out_hist.col = key.col;
                        out_hist.row1 = key.row1;
                        
                        double total_size = static_cast<double>(st.size); 
                        if (total_size < 1.0) total_size = 1.0; 

                        for (int b = 0; b < 10; ++b) {
                            out_hist.bins[b] = static_cast<float>(st.bins[b] / total_size);
                        }
                        local_buffer_hist.push_back(out_hist);
                    } 
                    else if (mode_flag == 0) {
                        double N = static_cast<double>(global_col_max_partitions[p][key.col]); 
                        
                        float skew_final = 0.0f;
                        float kurt_final = 0.0f;
                        float entropy = 10.0f; 
                        

                        if (N >= 4.0) {
                            double mean = st.s1 / N;
                            
                            double m2_raw = st.s2 / N;
                            double m3_raw = st.s3 / N;
                            double m4_raw = st.s4 / N;

                            double var = m2_raw - (mean * mean);

                            if (var > 1e-12) {
                                double mu3 = m3_raw - 3.0 * mean * m2_raw + 2.0 * (mean * mean * mean);
                                double mu4 = m4_raw - 4.0 * mean * m3_raw + 6.0 * (mean * mean) * m2_raw - 3.0 * (mean * mean * mean * mean);

                                double std_dev = std::sqrt(var);
                                double skew_pop = mu3 / (std_dev * std_dev * std_dev);
                                
                                skew_final = static_cast<float>((std::sqrt(N * (N - 1.0)) / (N - 2.0)) * skew_pop);

                                double kurt_pop_excess = (mu4 / (var * var)) - 3.0;
                                
                                kurt_final = static_cast<float>(((N - 1.0) / ((N - 2.0) * (N - 3.0))) * ((N + 1.0) * kurt_pop_excess + 6.0));
                            }
                        }

                        if (st.s1 > 1e-12 && std::isfinite(st.s1)) {
                            entropy = std::log(st.s1) - (st.raw_entropy / st.s1);
                        }

                        OutputEdge out;
                        out.row1 = key.row1;
                        out.col = key.col;
                        out.kurt = static_cast<float>(kurt_final);
                        out.skew = static_cast<float>(skew_final);
                        out.entropy = static_cast<float>(entropy);
                        
                        local_buffer_kurt.push_back(out);
                    }
                }
                
                std::lock_guard<std::mutex> lock(file_mtx);
                if (mode_flag == 2 && !local_buffer_hist.empty()) {
                    out_file.write(reinterpret_cast<const char*>(local_buffer_hist.data()), local_buffer_hist.size() * sizeof(OutputEdgeHist));
                } else if (mode_flag == 0 && !local_buffer_kurt.empty()) {
                    out_file.write(reinterpret_cast<const char*>(local_buffer_kurt.data()), local_buffer_kurt.size() * sizeof(OutputEdge));
                }
            });
        }
        for (auto& t : math_workers) { if (t.joinable()) t.join(); }
        out_file.close();
        
        std::cout << "Processamento (Acumulador Paralelo) concluído!" << std::endl;
    } else {
        std::cout << "Processamento (Writer) concluído! Resultados brutos guardados nos ficheiros das threads." << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::ofstream myFile(SaveAdress + "/time_" + seed + ".csv");
    myFile << elapsed.count();
    myFile.close();
    return 0;
}