#ifndef GRAPH_UTILS
#define GRAPH_UTILS

#include <string>
#include <random>

#include "graph.hpp"

namespace graph_utils{
    extern std::mt19937 prng;
    template<typename distribution,typename cmp>
    graph<cmp> gen_hypercubic(size_t,std::vector<size_t>,bool,distribution&);
    template<typename cmp>
    void save_graph(std::string,graph<cmp>&);
    template<typename cmp>
    graph<cmp> load_graph(std::string,size_t);
}

#endif
