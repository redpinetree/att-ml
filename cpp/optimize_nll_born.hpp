#ifndef OPTIMIZE_NLL_BORN_
#define OPTIMIZE_NLL_BORN_

#include <map>
#include <utility>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"

namespace optimize{
    template<typename cmp>
    double opt_struct_nll_born(graph<cmp>&,std::vector<sample_data>&,std::vector<int>&,std::vector<sample_data>&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    std::vector<int> classify_born(graph<cmp>&,std::vector<sample_data>&,std::vector<array1d<double> >&);
    
    template<typename cmp>
    void site_update_born(graph<cmp>&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array3d<double>&,array3d<double>&,int,double,double,double,double);
    template<typename cmp>
    array4d<double> fused_update_born(graph<cmp>&,bond&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::map<std::pair<int,int>,array4d<double> >&,std::map<std::pair<int,int>,array4d<double> >&,int,double,double,double,double);
    template<typename cmp>
    double calc_bmi_born(graph<cmp>&,bond&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
    double calc_ee_born(bond&,bond&);
};

#endif