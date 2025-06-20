/*
Copyright 2025 Katsuya O. Akamatsu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "observables.hpp"

std::vector<std::string> observables::output_lines;

void observables::write_output(std::string fn,std::vector<std::string>& lines){
    std::ofstream ofs(fn);
    for(int i=0;i<lines.size();i++){
        ofs<<lines[i];
    }
}

void observables::write_output(std::vector<std::string>& lines){
    for(int i=0;i<lines.size();i++){
        std::cout<<lines[i];
    }
}

void observables::write_binary_output(std::string fn,std::vector<std::pair<double,std::vector<double> > >& data){
    std::ofstream ofs(fn,std::ios::binary);
    int n_betas=data.size();
    ofs.write((char*) &n_betas,sizeof(n_betas));
    for(int n=0;n<data.size();n++){
        int data_size=data[n].second.size(); //for casting
        ofs.write((char*) &data[n].first,sizeof(data[n].first));
        ofs.write((char*) &data_size,sizeof(data[n].second.size()));
        for(int i=0;i<data[n].second.size();i++){
            ofs.write((char*) (&data[n].second[i]),sizeof(data[n].second[i]));
        }
    }
}

void observables::write_binary_output(std::vector<std::pair<double,std::vector<double> > >& data){
    for(int n=0;n<data.size();n++){
        std::cout<<"beta="<<data[n].first<<"\n";
        for(int i=0;i<data[n].second.size();i++){
            std::cout<<data[n].second[i]<<"\n";
        }
    }
}

/*
binary output format:
8b #blocks
[blocks]

block:
8b beta
8b #values
[values]
*/