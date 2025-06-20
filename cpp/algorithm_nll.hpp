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

#ifndef ALGORITHM_NLL_
#define ALGORITHM_NLL_

#include "graph.hpp"
#include "sampling.hpp"

namespace algorithm{
    std::vector<std::vector<array1d<double> > > load_data_from_file(std::string&,int&,int&,int&);
    std::vector<int> load_data_labels_from_file(std::string&,int&,int&);
    //nnTTN functions
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    void train_nll(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,int,int,bool,double,int,std::map<int,double>&,std::map<int,int>&,bool);
    
    //TTNBM functions
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);
    template<typename cmp>
    void train_nll_born(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,int,int,bool,double,int,std::map<int,double>&,std::map<int,int>&,bool);
    
    template<typename cmp>
    void calculate_site_probs(graph<cmp>&,bond&);
};

#endif
