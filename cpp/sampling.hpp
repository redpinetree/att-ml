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
    sample_data(size_t,std::vector<size_t>,double,double);
    size_t n_phys_sites() const;
    std::vector<size_t> s() const;
    double log_w() const;
    double e() const;
    size_t& n_phys_sites();
    std::vector<size_t>& s();
    double& log_w();
    double& e();
private:
    size_t n_phys_sites_;
    std::vector<size_t> s_;
    double log_w_;
    double e_;
};

namespace sampling{
    template<typename cmp>
    sample_data sample(graph<cmp>&);
    template<typename cmp>
    std::vector<sample_data> sample(graph<cmp>&,size_t);
    template<typename cmp>
    std::vector<sample_data> mh_sample(graph<cmp>&,size_t);
    template<typename cmp>
    std::vector<sample_data> mh_sample(graph<cmp>&,size_t,double&);
    std::vector<double> pair_overlaps(std::vector<sample_data>,size_t);
    std::vector<double> e_mc(std::vector<sample_data>&);
    std::vector<double> m_mc(std::vector<sample_data>&,size_t);
    std::vector<double> q_mc(std::vector<sample_data>&,size_t,std::vector<double>&);
    double expected_e(std::vector<sample_data>);
    double min_e(std::vector<sample_data>);
}

#endif
