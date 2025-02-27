#ifndef TTN_OPS_BORN_
#define TTN_OPS_BORN_

#include <utility>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"

template<typename cmp>
void canonicalize(graph<cmp>&,int);
template<typename cmp>
void canonicalize(graph<cmp>&);
// template<typename cmp>
// bool verify_canonical(graph<cmp>&);

template<typename cmp>
double calc_z_born(graph<cmp>&);
template<typename cmp>
std::vector<double > calc_w_born(graph<cmp>&,std::vector<sample_data>&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
template<typename cmp>
std::vector<double> update_cache_w_born(graph<cmp>&,int,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
std::vector<double> update_cache_w_born(bond&,bond&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);

#endif