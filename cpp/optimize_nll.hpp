/*
Copyright 2025 Katsuya O. Akamatsu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
    double opt_struct_nll(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    std::vector<int> classify(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,array2d<double>&);
    
    void site_update(bond&,double,std::vector<double>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,array3d<double>&,array3d<double>&,int,double,double,double,double);
    array4d<double> fused_update(bond&,bond&,double,std::vector<double>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::map<std::pair<int,int>,array4d<double> >&,std::map<std::pair<int,int>,array4d<double> >&,int,double,double,double,double);
    double calc_bmi(bond&,bond&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
    
    /* //TTNBM functions
    template<typename cmp>
    double opt_struct_nll_born(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    std::vector<int> classify_born(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<array1d<double> >&);
    
    template<typename cmp>
    void site_update_born(graph<cmp>&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array3d<double>&,array3d<double>&,int,double,double,double,double);
    template<typename cmp>
    array4d<double> fused_update_born(graph<cmp>&,bond&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::map<std::pair<int,int>,array4d<double> >&,std::map<std::pair<int,int>,array4d<double> >&,int,double,double,double,double);
    template<typename cmp>
    double calc_bmi_born(graph<cmp>&,bond&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
    double calc_ee_born(bond&,bond&); */
};

#endif