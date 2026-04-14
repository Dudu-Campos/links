#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <utility>
#include <string>

 std::vector<std::vector<int>> read_csv(std:: string Adress){
    

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


void write_csv(std::vector<std::vector<int>> vector, std:: string Adress){

  std:: ofstream MyFile(Adress);

    for (const auto& row : vector){
        for(int i=0; i<row.size();i++){
            if(i == 1){
                MyFile << ",";
            }
            MyFile << row[i];
        }
        MyFile << std::endl;

            
    }
    // Close the file
    MyFile.close();
}



void experiment(int number_iteration,std:: string reading_file,std:: string writing_file,int randomness,int walk_leng,int inicial_node){
    std::vector<std::vector<int>> dados;
    dados = read_csv(reading_file); 
    
    std:: vector<int> uniqueNodes;

    std::unordered_map<int,std::vector<int>> adj_list;

    for (const auto& row : dados) {
    int u = row[0];
    int v = row[1];
    if (!adj_list.count(u)) {
        uniqueNodes.push_back(u);
        adj_list[u] = std::vector<int>{};
    }
    if (!adj_list.count(v)) {
        uniqueNodes.push_back(v);
        adj_list[v] = std::vector<int>{};
    }
    adj_list[u].push_back(v);
    adj_list[v].push_back(u);
    }


    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<std::mt19937::result_type> distr(1,100);

    int currentNode = inicial_node;
    std:: vector<std:: vector<int>> walkPath;

    std::unordered_map<int,std::vector<int>> newNodes;
    
    for(int i = 0; i < number_iteration; i++){
        if(distr(engine) <= randomness){
            auto itAdj = adj_list.find(currentNode);
            size_t base = (itAdj == adj_list.end()) ? 0u : itAdj->second.size();
            auto itNew = newNodes.find(currentNode);
            size_t extra = (itNew == newNodes.end()) ? 0u : itNew->second.size();
            size_t total = base + extra;

            if (total == 0) {
                // sem vizinhos conhecidos -> teleport
                if (!uniqueNodes.empty()) {
                    std::uniform_int_distribution<std::mt19937::result_type> randomNodeSelector(0, uniqueNodes.size()-1);
                    int nextNode = uniqueNodes[randomNodeSelector(engine)];
                    std::vector<int> currentStep = {currentNode, nextNode};
                    walkPath.push_back(currentStep);
                    currentNode = nextNode;
                }
            } else {
                std::uniform_int_distribution<std::mt19937::result_type> nodedist(0, static_cast<int>(total - 1));
                int r = nodedist(engine);
                int nextNode;
                if (r < static_cast<int>(base)) {
                    nextNode = itAdj->second[r];
                } else {
                    nextNode = itNew->second[r - static_cast<int>(base)];
                }
                std::vector<int> currentStep = {currentNode, nextNode};
                walkPath.push_back(currentStep);
                currentNode = nextNode;
            }
        } else {
            // branch faltante: quando não escolhe vizinho, faça teleport (uniforme)
            if (!uniqueNodes.empty()) {
                std::uniform_int_distribution<std::mt19937::result_type> randomNodeSelector(0, uniqueNodes.size()-1);
                int nextNode = uniqueNodes[randomNodeSelector(engine)];
                std::vector<int> currentStep = {currentNode, nextNode};
                walkPath.push_back(currentStep);
                // opcional: registar em newNodes se quiser estudar novas ligações
                newNodes[currentNode].push_back(nextNode);
                newNodes[nextNode].push_back(currentNode);
                currentNode = nextNode;
            }
        }

        if (walkPath.size() % walk_leng == 0){
            currentNode = inicial_node;
            newNodes.clear();
        }
    }
    write_csv(walkPath,writing_file);
}

int main(){
    for(int i = 0;i<20;i++){
        std:: string wf = "/home/edu/Area_de_Trabalho/Projs/links/data/movies/testes/teste" + std:: to_string(i) + ".csv";
        experiment(1000*(100/(100-5*i)),"/home/edu/Area_de_Trabalho/Projs/links/data/movies/testes/grafo_0.csv",wf,i*5,50,107);
    }
    return 1;
}