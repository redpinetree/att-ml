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
    sample_data(int,std::vector<int>);
    int n_phys_sites() const;
    std::vector<int> s() const;
    int& n_phys_sites();
    std::vector<int>& s();
private:
    int n_phys_sites_;
    std::vector<int> s_;
};

namespace sampling{
    template<typename cmp>
    std::vector<sample_data> tree_sample(int,graph<cmp>&,int);
    template<typename cmp>
    std::vector<sample_data> tree_sample(graph<cmp>&,int);
}

#endif
