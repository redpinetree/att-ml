#ifndef ALGORITHM
#define ALGORITHM

#include "../ndarray.hpp"
#include "../graph.hpp"
#include "../site.hpp"
#include "../sampling.hpp"

namespace algorithm{
    template<typename cmp>
    void approx(size_t,graph<cmp>&,size_t,size_t,std::string,std::string,size_t);
    template<typename cmp>
    void calculate_site_probs(graph<cmp>&,bond&);
};

#endif
