#ifndef SAMPLING
#define SAMPLING

#include <string>
#include <complex>
#include <vector>
#include <tuple>
#include <map>

#include "graph.hpp"
#include "ndarray.hpp"

namespace sampling{
    template<typename cmp>
    std::vector<std::vector<size_t> > sample(graph<cmp>&,size_t);
    std::vector<double> pair_overlaps(std::vector<std::vector<size_t> >,size_t);
    
    void write_output(std::string,std::vector<double>&,double);
    void write_output(std::vector<double>&);
}

#endif
