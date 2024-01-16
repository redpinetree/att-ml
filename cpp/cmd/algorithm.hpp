#ifndef ALGORITHM
#define ALGORITHM

#include "../ndarray.hpp"
#include "../graph.hpp"
#include "../site.hpp"

namespace algorithm{
    template<typename cmp>
    void approx(size_t,graph<cmp>&,size_t,size_t,double);
    template<typename cmp>
    void calculate_site_probs(graph<cmp>&,bond&);
};

#endif
