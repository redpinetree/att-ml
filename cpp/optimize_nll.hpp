#ifndef OPTIMIZE_NLL_
#define OPTIMIZE_NLL_

#include <utility>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"

namespace optimize{
    template<typename cmp>
    double opt_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,size_t);
    template<typename cmp>
    double opt_struct_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t);
    template<typename cmp>
    double hopt_nll(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    double hopt_nll2(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    std::vector<size_t> classify(graph<cmp>&,std::vector<sample_data>&,std::vector<array1d<double> >&);
    
    void site_update(bond&,double,std::vector<double>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array3d<double>&,array3d<double>&,size_t,double,double,double,double);
    array4d<double> fused_update(bond&,bond&,double,std::vector<double>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,size_t,double,double,double,double);
    double calc_bmi(bond&,bond&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
    template<typename cmp>
    double test_connection_scheme(graph<cmp>&,array4d<double>&,bond&,bond&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
};

#endif