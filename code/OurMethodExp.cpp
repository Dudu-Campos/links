#include <iostream>
#include <random>
#include <cstdlib>
#include <string>
#include <chrono>
#include <ctime>
#include <iterator> 
#include <cassert>
#include "prob_calc.h"
#include <filesystem>
#include <sys/statvfs.h> 
#include <future>
#include <mutex>
#include <queue>
#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
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
    float bins[10]; // As 10 "fatias" da distribuição (Decis)
};

struct TaskDef {
    int col;
    int fixed_idx;
    int peso; // -1 significa "calcular tudo". >= 0 significa "tarefa fatiada"
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
    bool use_writer = (mode_flag == 1);

    std::filesystem::create_directories(SaveAdress);

    Eigen::SparseMatrix<bool, Eigen::ColMajor> B;
    std::unordered_map<int,int> left_id_map, right_id_map, inverse_left, inverse_right;
    std::unordered_map<int,int> left_degree, right_degree;
    
    auto data = read_csv(TrainAdress);
    bool is_directed = false;
    
    build_bipartite(data, B, left_id_map, right_id_map, inverse_left, inverse_right, left_degree, right_degree, is_directed);
  
    // ==============================================================
    // PRÉ-CÁLCULO DA TABELA DE LOGARITMOS (Itens / Direita)
    // ==============================================================
    std::vector<double> inv_log_right(inverse_right.size(), 1.0);
    double eps = std::numeric_limits<double>::epsilon();

    for (int k = 0; k < (int)inverse_right.size(); ++k) {
        int original_v = inverse_right[k]; 
        int d_right = right_degree[original_v]; 
        
        if (d_right > 0) {
            inv_log_right[k] =  std::log(1.0 + d_right + eps);
        }
    }   
   
    std::cout << "left_id_map size: " << left_id_map.size() << std::endl;
    std::cout << "right_id_map size: " << right_id_map.size() << std::endl;

    std::vector<TaskDef> tasks;
    std::atomic<size_t> current_task{0};
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 12;

    std::vector<std::thread> workers;
    const int HUB_THRESHOLD = 40;
    for (int col = 0; col < left_id_map.size() ; ++col) {
        int deg = left_degree[inverse_left[col]];
        
        if (deg > HUB_THRESHOLD) {
            for (int j = 0; j < deg; ++j) {
                tasks.push_back({col, j,deg});
            }
        } else {
            tasks.push_back({col, -1,deg}); 
        }
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::unordered_map<EdgeKey, StatsAccumulator, EdgeKeyHash>> all_edge_stats(num_threads);
    std::vector<std::unordered_map<int32_t, int>> all_col_max_size(num_threads);

    std::sort(tasks.begin(), tasks.end(), [](const TaskDef& a, const TaskDef& b) {
        if (a.peso != b.peso) return a.peso > b.peso; 
        return a.fixed_idx < b.fixed_idx; 
    });

    // ===================== WORKERS =====================
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
                        inv_log_right, local_pipe
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
    
    // ========================================================================
    // AGREGAÇÃO E ESCRITA FINAL (MODOS 0 e 2)
    // ========================================================================
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

        // --- FASE 1: MERGE (Fundir os mapas) ---
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

                            if (prob >= 0.5f) {
                                bin_index = 9;       // Probabilidades gigantes (50% a 100%)
                            } else if (prob >= 0.5f) {
                                bin_index = 8;       // 20% a 50%
                            } else if (prob >= 0.35f) {
                                bin_index = 7;       // 10% a 20%
                            } else if (prob >= 0.25f) {
                                bin_index = 6;       // 5% a 10%
                            } else if (prob >= 0.20f) {
                                bin_index = 5;       // 1% a 5%
                            } else if (prob >= 0.15f) {
                                bin_index = 4;       // 0.5% a 1%
                            } else if (prob >= 0.10f) {
                                bin_index = 3;       // 0.1% a 0.5%
                            } else if (prob >= 0.05f) {
                                bin_index = 2;       // 0.05% a 0.1%
                            } else if (prob >= 0.01f) {
                                bin_index = 1;       // 0.01% a 0.05%
                            } else {
                                bin_index = 0;       // Probabilidades microscópicas (< 0.01%)
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

        // --- FASE 2: MATEMÁTICA E ESCRITA ---
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
                        // MODO 2: HISTOGRAMAS (10 DECIS)
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
                        // MODO 0: CURTOSE E SKEWNESS
                        double N = static_cast<double>(global_col_max_partitions[p][key.col]); 
                        double skew = 0.0;
                        double kurt_acoplada = 0.0;
                        double entropy = 10.0;

                        if (N >= 4.0) {
                            double mu = st.s1 / N;
                            double m2 = st.s2 - N * (mu * mu);
                            double m3 = st.s3 - 3.0 * mu * st.s2 + 3.0 * (mu * mu) * st.s1 - N * (mu * mu * mu);
                            double var = m2 / (N - 1.0);

                            if (var > 1e-12) {
                                double std_dev = std::sqrt(var);
                                skew = (N / ((N - 1.0) * (N - 2.0))) * (m3 / (std_dev * std_dev * std_dev));
                            }

                            double N_acoplado = N + 1.0;
                            double s1_acoplado = st.s1 + skew;
                            double s2_acoplado = st.s2 + (skew * skew);
                            double s3_acoplado = st.s3 + (skew * skew * skew);
                            double s4_acoplado = st.s4 + (skew * skew * skew * skew);

                            double mu_acop = s1_acoplado / N_acoplado;
                            double m2_acop = s2_acoplado - N_acoplado * (mu_acop * mu_acop);
                            
                            double m4_acop = s4_acoplado 
                                        - 4.0 * mu_acop * s3_acoplado 
                                        + 6.0 * (mu_acop * mu_acop) * s2_acoplado 
                                        - 4.0 * (mu_acop * mu_acop * mu_acop) * s1_acoplado 
                                        + N_acoplado * (mu_acop * mu_acop * mu_acop * mu_acop);

                            double var_acop = m2_acop / (N_acoplado - 1.0);

                            if (var_acop > 1e-12 && std::isfinite(var_acop)) {
                                double std_acop = std::sqrt(var_acop);
                                double term1 = (N_acoplado * (N_acoplado + 1.0)) / ((N_acoplado - 1.0) * (N_acoplado - 2.0) * (N_acoplado - 3.0));
                                double denom = (std_acop * std_acop * std_acop * std_acop);
                                if (denom < 1e-12) denom = 1e-12;
                                double term2 = m4_acop / denom;
                                double term3 = (3.0 * (N_acoplado - 1.0) * (N_acoplado - 1.0)) / ((N_acoplado - 2.0) * (N_acoplado - 3.0));

                                kurt_acoplada = (term1 * term2) - term3;
                            }
                        }

                        if (st.s1 > 1e-12 && std::isfinite(st.s1)) {
                            entropy = std::log(st.s1) - (st.raw_entropy / st.s1);
                        }

                        OutputEdge out;
                        out.row1 = key.row1;
                        out.col = key.col;
                        out.kurt_skew = static_cast<float>(kurt_acoplada * skew);
                        out.entropy = static_cast<float>(entropy);
                        
                        local_buffer_kurt.push_back(out);
                    }
                }
                
                // ESCRITA PROTEGIDA POR LOCK
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

    // Gravação do tempo no final, independentemente do modo escolhido
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::ofstream myFile(SaveAdress + "/time_" + seed + ".csv");
    myFile << elapsed.count();
    myFile.close();

    return 0;
}