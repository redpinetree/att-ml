#ifndef OPTIMIZE_NLL_
#define OPTIMIZE_NLL_

#include <utility>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"

namespace optimize{
    template<typename cmp>
    double opt_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,size_t);
    template<typename cmp>
    double hopt_nll(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    double hopt_nll2(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    std::vector<size_t> classify(graph<cmp>&,std::vector<sample_data>&,std::vector<array1d<double> >&);
};

#endif