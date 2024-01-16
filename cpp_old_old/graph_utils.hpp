#ifndef GRAPH_UTILS
#define GRAPH_UTILS

#include <string>
#include <random>

#include "graph.hpp"

namespace graph_utils{
    extern std::mt19937 prng;
    template <typename distribution>
    graph_old gen_hypercubic_old(size_t,std::vector<size_t>,bool,distribution&);
    void save_graph_old(std::string,graph_old&);
    graph_old load_graph_old(std::string,size_t);
}

#endif
