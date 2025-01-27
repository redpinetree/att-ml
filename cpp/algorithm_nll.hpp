#ifndef ALGORITHM_NLL_
#define ALGORITHM_NLL_

#include "graph.hpp"
#include "sampling.hpp"

namespace algorithm{
    std::vector<sample_data> load_data_from_file(std::string&,size_t&,size_t&,size_t&);
    std::vector<size_t> load_data_labels_from_file(std::string&,size_t,size_t&);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    
#ifdef MODEL_TREE_ML_BORN
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<sample_data>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
#endif
    
    template<typename cmp>
    void calculate_site_probs(graph<cmp>&,bond&);
};

#endif
