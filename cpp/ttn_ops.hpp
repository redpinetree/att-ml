#ifndef TTN_OPS_
#define TTN_OPS_

#include <set>
#include <utility>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"

void normalize(array3d<double>&);
void normalize(array4d<double>&);
void normalize_using_z(array3d<double>&,double);
void normalize_using_z(array4d<double>&,double);

std::vector<array3d<double> > calc_dz(std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);
template<typename cmp>
double calc_z(graph<cmp>&);
template<typename cmp>
double calc_z(graph<cmp>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);
std::vector<std::vector<array3d<double> > > calc_dw(std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
template<typename cmp>
double update_cache_z(graph<cmp>&,size_t,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);
std::vector<std::vector<array3d<double> > > calc_dw(std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
template<typename cmp>
std::vector<double > calc_w(graph<cmp>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);
template<typename cmp>
std::vector<double> update_cache_w(graph<cmp>&,size_t,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

#endif