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
