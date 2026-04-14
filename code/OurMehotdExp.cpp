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
#include <algorithm>

bool has_enough_disk_space(const std::string& filepath, double min_free_fraction = 0.10) {
    // get directory part of filepath
    std::string dir = filepath;
    auto pos = dir.rfind('/');
    if (pos == std::string::npos) dir = ".";
    else if (pos == 0) dir = "/"; 
    else dir = dir.substr(0, pos);

    struct statvfs st;
    if (statvfs(dir.c_str(), &st) != 0) {
        std::cerr << "statvfs failed for " << dir << " (" << strerror(errno) << ")\n";
        return false;
    }

    unsigned long long total = static_cast<unsigned long long>(st.f_blocks) * st.f_frsize;
    unsigned long long avail = static_cast<unsigned long long>(st.f_bavail) * st.f_frsize;

    if (total == 0) return false;
    double free_fraction = static_cast<double>(avail) / static_cast<double>(total);
    return free_fraction > min_free_fraction;
}


int main(int argc, char* argv[]){

    std::string Adress;
    std::string SaveAdress;
    int NeighSize = 1;
    int PrunningRatio = 1;
    std::string reverse;
    if (argc > 1) {
        Adress = argv[1];
    }
    if (argc > 2) {
        SaveAdress = argv[2];
    }
    if (argc > 3) {
        NeighSize = std::stoi(argv[3]);
    }
    if (argc > 4) {
        PrunningRatio = std::stoi(argv[4]);
    }
    if (argc > 5) {
        reverse = (argv[5]);
    }

 
    auto edges = read_csv(Adress);
    std:: unordered_map<int,int> degreeCounter;
    if (reverse == "inverse") {
        for (auto &edge : edges) {
            if (edge.size() >= 2) std::swap(edge[0], edge[1]);
        }
    }
        int numberEdges = 0;
    for(int i=0;i<edges.size();i++){
        if(degreeCounter.count(edges[i][0])){
            degreeCounter[edges[i][0]]++;
        }
        else{
            degreeCounter[edges[i][0]] = 1;
            numberEdges++;
        }
    }
    
    std::clock_t start;
    double duration;
    start = std::clock();  
    
    int cont=0;
    
    Eigen::SparseMatrix<bool, Eigen::ColMajor> B;
    std::unordered_map<int,int> right_id_map,left_id_map,inverse_right,inverse_left,left_degree,right_degree;
    build_bipartite(edges, B, left_id_map, right_id_map, inverse_left, inverse_right,left_degree,right_degree);
    
    std::vector<std::vector<double>> times;
    std::string out_dir = SaveAdress;
    std::filesystem::create_directories(out_dir); // ensure dir exists before loop

    // debug: show how many unique left nodes we will process
    {
        // std::lock_guard<std::mutex> lk(io_mutex);
        std::cout << "Unique left nodes to process: " << degreeCounter.size() << std::endl;
    }

    // --- thread-pool with semaphore to limit concurrency ---
    std::mutex io_mutex;
    unsigned max_threads = std::thread::hardware_concurrency();
    // if (max_threads == 0) 
    max_threads = 12;

    std::condition_variable cv;
    std::mutex cv_m;
    std::atomic<int> active{0};
    std::vector<std::thread> threads;
    threads.reserve(degreeCounter.size());

    for (auto const& pair : degreeCounter) {
        // wait until there's capacity
        {
            std::unique_lock<std::mutex> lk(cv_m);
            cv.wait(lk, [&]{ return active.load() < static_cast<int>(max_threads); });
            active.fetch_add(1);
        }

        threads.emplace_back([pair, &B, &left_id_map, &inverse_left, &inverse_right,&left_degree,&right_degree, &out_dir, NeighSize,&io_mutex, &active, &cv]() {
            // RAII guard to ensure active-- and notify on any exit
            struct ExitNotify { std::atomic<int>& active; std::condition_variable& cv; ExitNotify(std::atomic<int>& a, std::condition_variable& c): active(a), cv(c){} ~ExitNotify(){ active.fetch_sub(1); cv.notify_all();; } };
            ExitNotify on_exit(active, cv);

            try {
                auto start = std::chrono::steady_clock::now();

                int left_orig = pair.first;
                auto it = left_id_map.find(left_orig);
                if (it == left_id_map.end()) {
                    std::lock_guard<std::mutex> lok(io_mutex);
                    std::cerr << "Warning: left node not found in left_id_map: " << left_orig << "\n";
                    return;
                }
                int left_idx = it->second;

                {
                    std::lock_guard<std::mutex> lok(io_mutex);
                    // std::cout << "left_orig=" << left_orig << " left_idx=" << left_idx << std::endl;
                }

                std::vector<std::vector<double>> result = calc_probability(B, left_idx, NeighSize, inverse_left, inverse_right,left_degree,right_degree);

                auto end = std::chrono::steady_clock::now();


                std::string adress = out_dir + "/" + std::to_string(NeighSize) + "_" + std::to_string(left_orig) + ".csv";
                if(has_enough_disk_space(adress)){
                    write_csv(result, adress);
                    result.clear();
                    result.shrink_to_fit();
                }
            }
            catch (const std::exception &ex) {
                std::lock_guard<std::mutex> lok(io_mutex);
                std::cerr << "Exception processing left " << pair.first << ": " << ex.what() << "\n";
            }
            catch (...) {
                std::lock_guard<std::mutex> lok(io_mutex);
                std::cerr << "Unknown exception processing left " << pair.first << "\n";
            }
        });
    }

    // join all threads
    for (auto &t : threads) if (t.joinable()) t.join();
    times = {{( std::clock() - start ) / (double) CLOCKS_PER_SEC}};
    // write times after all tasks finished
    std::string TimeAdress = out_dir  + "/exectime2.csv";
    write_csv(times, TimeAdress);
}