#ifndef ALGORITHM_NLL_
#define ALGORITHM_NLL_

#include "graph.hpp"
#include "sampling.hpp"

namespace algorithm{
    std::vector<sample_data> load_data_from_file(std::string&,size_t&,size_t&,size_t&);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<sample_data>&,size_t);
    template<typename cmp>
    void train_nll(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    void calculate_site_probs(graph<cmp>&,bond&);
};

#endif
