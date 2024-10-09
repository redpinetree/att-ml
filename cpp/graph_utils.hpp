#ifndef GRAPH_UTILS
#define GRAPH_UTILS

#include <string>
#include <random>

#include "graph.hpp"

namespace graph_utils{
    template<typename distribution,typename cmp>
    graph<cmp> gen_hypercubic(size_t,std::vector<size_t>,bool,distribution&,double);
    template<typename cmp>
    graph<cmp> load_graph(std::string,size_t,double);
    template<typename distribution,typename cmp>
    graph<cmp> init_pbttn(size_t,size_t,size_t,std::vector<size_t>,distribution&,double);
    template<typename distribution,typename cmp>
    graph<cmp> init_pbttn(size_t,size_t,std::vector<size_t>,distribution&,double);
    template<typename distribution,typename cmp>
    graph<cmp> init_mps(size_t,size_t,size_t,std::vector<size_t>,distribution&,double);
    template<typename distribution,typename cmp>
    graph<cmp> init_mps(size_t,size_t,std::vector<size_t>,distribution&,double);
    template<typename distribution,typename cmp>
    graph<cmp> init_rand(size_t,size_t,size_t,std::vector<size_t>,distribution&,double);
    template<typename distribution,typename cmp>
    graph<cmp> init_rand(size_t,size_t,std::vector<size_t>,distribution&,double);
}

#endif
