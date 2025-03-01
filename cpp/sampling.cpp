#include <iostream>
#include <fstream>
#include <queue>
#include <random>
#include <tuple>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"
#include "ttn_ops.hpp"
#include "utils.hpp"

sample_data::sample_data(){}
sample_data::sample_data(int n_phys_sites,std::vector<int> s):n_phys_sites_(n_phys_sites),s_(s){}
int sample_data::n_phys_sites() const{return this->n_phys_sites_;}
std::vector<int> sample_data::s() const{return this->s_;}
int& sample_data::n_phys_sites(){return this->n_phys_sites_;}
std::vector<int>& sample_data::s(){return this->s_;}

template<typename cmp>
std::vector<sample_data> sampling::tree_sample(int root,graph<cmp>& g,int n_samples){
    std::queue<int> todo_idxs;
    std::discrete_distribution<int> pdf(g.vs()[root].p_k().begin(),g.vs()[root].p_k().end());
    std::vector<sample_data> s_vec;
    for(int n=0;n<n_samples;n++){
        s_vec.push_back(sample_data(g.n_phys_sites(),std::vector<int>(g.vs().size(),0)));
        s_vec[n].s()[root]=pdf(prng)+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
    }
    if(g.vs()[root].virt()){
        todo_idxs.push(root);
    }
    while(!todo_idxs.empty()){
        int idx=todo_idxs.front();
        todo_idxs.pop();
        int v1=g.vs()[idx].l_idx();
        int v2=g.vs()[idx].r_idx();
        if(g.vs()[v1].virt()){
            todo_idxs.push(v1);
        }
        if(g.vs()[v2].virt()){
            todo_idxs.push(v2);
        }
        std::vector<std::discrete_distribution<int> > cond_prob_dists;
        // std::cout<<(std::string)g.vs()[idx].p_ijk()<<"\n";
        // for(int i=0;i<g.vs()[idx].p_k().size();i++){
            // std::cout<<g.vs()[idx].p_k()[i]<<" ";
        // }
        // std::cout<<"\n";
        for(int k=0;k<g.vs()[idx].rank();k++){
            std::vector<double> cond_probs;
            for(int i=0;i<g.vs()[v1].rank();i++){
                for(int j=0;j<g.vs()[v2].rank();j++){
                    cond_probs.push_back(g.vs()[idx].p_ijk().at(i,j,k)); //no need to normalize, handled by discrete_distribution
                }
            }
            cond_prob_dists.push_back(std::discrete_distribution<int>(cond_probs.begin(),cond_probs.end()));
        }
        for(int n=0;n<n_samples;n++){
            if(s_vec[n].s()[idx]!=0){
                pdf=cond_prob_dists[s_vec[n].s()[idx]-1];
                int composite_idx=pdf(prng);
                s_vec[n].s()[v1]=(composite_idx/g.vs()[v2].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
                s_vec[n].s()[v2]=(composite_idx%g.vs()[v2].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
            }
        }
    }
    return s_vec;
}
template std::vector<sample_data> sampling::tree_sample(int,graph<bmi_comparator>&,int);

template<typename cmp>
std::vector<sample_data> sampling::tree_sample(graph<cmp>& g,int n_samples){
    std::vector<sample_data> s_vec=sampling::tree_sample(g.vs().size()-1,g,n_samples);
    return s_vec;
}
template std::vector<sample_data> sampling::tree_sample(graph<bmi_comparator>&,int);
