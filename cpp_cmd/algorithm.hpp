#ifndef ALGORITHM
#define ALGORITHM

#include "ndarray.hpp"
#include "graph.hpp"
#include "site.hpp"

namespace algorithm{
    template<typename cmp>
    void cmd_approx(size_t,graph<cmp>&,size_t);
};

#endif
