#ifndef GRAPH_UTILS
#define GRAPH_UTILS

#include <vector>

#include "graph.hpp"

namespace graph_utils{
    template<typename cmp>
    graph<cmp> init_pbttn(int,int,int,int);
    template<typename cmp>
    graph<cmp> init_pbttn(int,int,int);
    template<typename cmp>
    graph<cmp> init_mps(int,int,int,int);
    template<typename cmp>
    graph<cmp> init_mps(int,int,int);
    template<typename cmp>
    graph<cmp> init_rand(int,int,int,int);
    template<typename cmp>
    graph<cmp> init_rand(int,int,int);
}

#endif
