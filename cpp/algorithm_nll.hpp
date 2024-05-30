#ifndef ALGORITHM_NLL_
#define ALGORITHM_NLL_

#include "graph.hpp"

namespace algorithm{
    template<typename cmp>
    std::vector<double> train_nll(graph<cmp>&,size_t,size_t,size_t);
};

#endif
