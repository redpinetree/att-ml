#ifndef SAMPLING
#define SAMPLING

#include <string>
#include <complex>
#include <vector>
#include <tuple>
#include <map>

#include "graph.hpp"
#include "ndarray.hpp"

class sample_data{
public:
    sample_data();
    sample_data(size_t,std::vector<size_t>);
    size_t n_phys_sites() const;
    std::vector<size_t> s() const;
    size_t& n_phys_sites();
    std::vector<size_t>& s();
private:
    size_t n_phys_sites_;
    std::vector<size_t> s_;
};

namespace sampling{
    template<typename cmp>
    std::vector<sample_data> tree_sample(size_t,graph<cmp>&,size_t);
    template<typename cmp>
    std::vector<sample_data> tree_sample(graph<cmp>&,size_t);
}

#endif
