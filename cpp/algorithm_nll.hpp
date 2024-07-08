#ifndef ALGORITHM_NLL_
#define ALGORITHM_NLL_

#include "graph.hpp"

namespace algorithm{
    template<typename cmp>
    void train_nll(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    void calculate_site_probs(graph<cmp>&,bond&);
};

#endif
