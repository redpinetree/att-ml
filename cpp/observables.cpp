#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "observables.hpp"

std::vector<std::string> observables::output_lines;

void observables::write_output(std::string fn,std::vector<std::string>& lines){
    std::ofstream ofs(fn);
    for(size_t i=0;i<lines.size();i++){
        ofs<<lines[i];
    }
}

void observables::write_output(std::vector<std::string>& lines){
    for(size_t i=0;i<lines.size();i++){
        std::cout<<lines[i];
    }
}

void observables::write_binary_output(std::string fn,std::vector<std::pair<double,std::vector<double> > >& data){
    std::ofstream ofs(fn,std::ios::binary);
    size_t n_betas=data.size();
    ofs.write((char*) &n_betas,sizeof(n_betas));
    for(size_t n=0;n<data.size();n++){
        size_t data_size=data[n].second.size(); //for casting
        ofs.write((char*) &data[n].first,sizeof(data[n].first));
        ofs.write((char*) &data_size,sizeof(data[n].second.size()));
        for(size_t i=0;i<data[n].second.size();i++){
            ofs.write((char*) (&data[n].second[i]),sizeof(data[n].second[i]));
        }
    }
}

void observables::write_binary_output(std::vector<std::pair<double,std::vector<double> > >& data){
    for(size_t n=0;n<data.size();n++){
        std::cout<<"beta="<<data[n].first<<"\n";
        for(size_t i=0;i<data[n].second.size();i++){
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