#include <iostream>
#include <fstream>
#include <queue>
#include <random>
#include <vector>

#include "mpi_utils.hpp"
#include "ndarray.hpp"
#include "sampling.hpp"
#include "utils.hpp"

template<typename cmp>
std::vector<std::vector<size_t> > sampling::sample(graph<cmp>& g,size_t n_samples){
    std::vector<std::vector<size_t> > samples;
    for(size_t n=0;n<n_samples;n++){
        std::vector<size_t> sample(g.vs().size(),0);
        std::queue<size_t> todo_idxs;
        std::discrete_distribution<size_t> pdf(g.vs()[g.vs().size()-1].p_k().begin(),g.vs()[g.vs().size()-1].p_k().end());
        sample[g.vs().size()-1]=pdf(mpi_utils::prng);
        if(g.vs()[g.vs().size()-1].virt()){
            todo_idxs.push(g.vs().size()-1);
        }
        while(!todo_idxs.empty()){
            size_t idx=todo_idxs.front();
            todo_idxs.pop();
            size_t v1=g.vs()[idx].p().first;
            size_t v2=g.vs()[idx].p().second;
            if(g.vs()[v1].virt()){
                todo_idxs.push(v1);
            }
            if(g.vs()[v2].virt()){
                todo_idxs.push(v2);
            }
            std::vector<double> cond_probs;
            for(size_t i=0;i<g.vs()[v1].rank();i++){
                for(size_t j=0;j<g.vs()[v2].rank();j++){
                    cond_probs.push_back(g.vs()[idx].p_ijk().at(i,j,sample[idx])/g.vs()[idx].p_k()[sample[idx]]);
                }
            }
            pdf=std::discrete_distribution<size_t>(cond_probs.begin(),cond_probs.end());
            size_t composite_idx=pdf(mpi_utils::prng);
            sample[v1]=composite_idx/g.vs()[v2].rank();
            sample[v2]=composite_idx%g.vs()[v2].rank();
        }
        sample.resize(g.n_phys_sites());
        // for(size_t m=0;m<sample.size();m++){
            // std::cout<<sample[m]<<" ";
        // }
        // std::cout<<"\n";
        samples.push_back(sample);
    }
    return samples;
}
template std::vector<std::vector<size_t> > sampling::sample(graph<bmi_comparator>&,size_t);

std::vector<double> sampling::pair_overlaps(std::vector<std::vector<size_t> > samples,size_t q){
    std::vector<double> overlaps;
    for(size_t n1=0;n1<samples.size();n1++){
        for(size_t n2=n1+1;n2<samples.size();n2++){
            double overlap=0;
            for(size_t m=0;m<samples[n1].size();m++){
                overlap+=(samples[n1][m]==samples[n2][m]?1:-1/(double)(q-1));
            }
            overlap/=samples[n1].size();
            overlaps.push_back(overlap);
        }
    }
    // for(size_t m=0;m<overlaps.size();m++){
        // std::cout<<overlaps[m]<<" ";
    // }
    // std::cout<<"\n";
    return overlaps;
}

void sampling::write_output(std::string fn,std::vector<double>& data,double beta){
    std::ofstream ofs(fn,std::ios::app|std::ios::binary);
    size_t data_size=data.size(); //for casting
    ofs.write((char*) &beta,sizeof(beta));
    ofs.write((char*) &data_size,sizeof(data.size()));
    for(size_t i=0;i<data.size();i++){
        ofs.write((char*) (&data[i]),sizeof(data[i]));
    }
}

void sampling::write_output(std::vector<double>& data){
    for(size_t i=0;i<data.size();i++){
        std::cout<<data[i];
    }
}

/*
overlap output format:
8b #blocks
[blocks]

block:
8b beta
8b #values
[values]
*/