#ifndef ALGORITHM
#define ALGORITHM

#include "ndarray.hpp"
#include "graph.hpp"
#include "site.hpp"

namespace algorithm{
    size_t f(site,size_t,site,size_t);
    array2d<size_t> f_maxent(site,site,array2d<double>,size_t);
    template<typename cmp>
    void cmd_approx(size_t,graph<cmp>&,size_t);
};

#endif
