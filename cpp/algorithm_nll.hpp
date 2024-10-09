#ifndef ALGORITHM_NLL_
#define ALGORITHM_NLL_

#include "graph.hpp"
#include "sampling.hpp"

namespace algorithm{
    std::vector<sample_data> load_training_data_from_file(std::string&,size_t&,size_t&,size_t&);
    std::vector<size_t> load_training_data_labels_from_file(std::string&,size_t,size_t&);
    template<typename cmp>
    void train_nll(graph<cmp>&,size_t,size_t,size_t,double);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,std::vector<double>&);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<sample_data>&,size_t,size_t,bool,double,std::vector<double>&);
    template<typename cmp>
    void calculate_site_probs(graph<cmp>&,bond&);
};

#endif
