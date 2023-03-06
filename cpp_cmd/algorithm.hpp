#ifndef ALGORITHM
#define ALGORITHM

#include "graph.hpp"
#include "site.hpp"

namespace algorithm{
    size_t f(site,size_t,site,size_t);
    template<typename cmp>
    void cmd_approx(size_t,graph<cmp>&);
};

#endif
