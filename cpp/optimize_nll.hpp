#ifndef OPTIMIZE_NLL_
#define OPTIMIZE_NLL_

#include <map>
#include <utility>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"

namespace optimize{
    //nnTTN functions
    template<typename cmp>
    double opt_struct_nll(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    std::vector<size_t> classify(graph<cmp>&,std::vector<sample_data>&,std::vector<array1d<double> >&);
    
    void site_update(bond&,double,std::vector<double>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array3d<double>&,array3d<double>&,size_t,double,double,double,double);
    array4d<double> fused_update(bond&,bond&,double,std::vector<double>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,size_t,double,double,double,double);
    double calc_bmi(bond&,bond&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
    
    //TTNBM functions
    template<typename cmp>
    double opt_struct_nll_born(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
    template<typename cmp>
    std::vector<size_t> classify_born(graph<cmp>&,std::vector<sample_data>&,std::vector<array1d<double> >&);
    
    template<typename cmp>
    void site_update_born(graph<cmp>&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array3d<double>&,array3d<double>&,size_t,double,double,double,double);
    template<typename cmp>
    array4d<double> fused_update_born(graph<cmp>&,bond&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,size_t,double,double,double,double);
    template<typename cmp>
    double calc_bmi_born(graph<cmp>&,bond&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
    double calc_ee_born(bond&,bond&);
};

#endif