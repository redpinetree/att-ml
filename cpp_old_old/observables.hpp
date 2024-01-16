#ifndef OBSERVABLES
#define OBSERVABLES

#include <string>
#include <vector>
#include <tuple>
#include <map>

#include "graph.hpp"
#include "ndarray.hpp"

namespace observables{
    extern std::vector<std::string> output_lines;
    extern std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t> >,double> known_factors;
    template<typename cmp>
    graph<cmp> cmd_treeify(graph<cmp>&);
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,size_t,size_t);
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<size_t>,size_t); //top-down
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,std::vector<size_t>); //bottom-up
    template<typename cmp>
    void print_moments(graph<cmp>&,size_t,size_t); //debug
    void write_output(std::string,std::vector<std::string>&);
    void write_output(std::vector<std::string>&);
    
    
}

#endif
