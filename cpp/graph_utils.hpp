#ifndef GRAPH_UTILS
#define GRAPH_UTILS

#include <vector>

#include "graph.hpp"

namespace graph_utils{
    template<typename cmp>
    graph<cmp> init_pbttn(size_t,size_t,size_t,std::vector<size_t>);
    template<typename cmp>
    graph<cmp> init_pbttn(size_t,size_t,std::vector<size_t>);
    template<typename cmp>
    graph<cmp> init_mps(size_t,size_t,size_t,std::vector<size_t>);
    template<typename cmp>
    graph<cmp> init_mps(size_t,size_t,std::vector<size_t>);
    template<typename cmp>
    graph<cmp> init_rand(size_t,size_t,size_t,std::vector<size_t>);
    template<typename cmp>
    graph<cmp> init_rand(size_t,size_t,std::vector<size_t>);
}

#endif
