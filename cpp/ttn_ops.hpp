#ifndef TTN_OPS_
#define TTN_OPS_

#include <utility>
#include <vector>

#include "ndarray.hpp"
#include "sampling.hpp"

//nnTTN functions
void normalize(array3d<double>&);
void normalize(array4d<double>&);
void normalize_using_z(array3d<double>&,double);
void normalize_using_z(array4d<double>&,double);

std::vector<array3d<double> > calc_dz(std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);
template<typename cmp>
double calc_z(graph<cmp>&);
template<typename cmp>
double calc_z(graph<cmp>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);
std::vector<std::vector<array3d<double> > > calc_dw(std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
template<typename cmp>
double update_cache_z(graph<cmp>&,int,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);
std::vector<std::vector<array3d<double> > > calc_dw(std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
template<typename cmp>
std::vector<double > calc_w(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
template<typename cmp>
std::vector<double> update_cache_w(graph<cmp>&,int,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
double update_cache_z(bond&,bond&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);
std::vector<double> update_cache_w(bond&,bond&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);

/* //TTNBM functions
template<typename cmp>
void canonicalize(graph<cmp>&,int);
template<typename cmp>
void canonicalize(graph<cmp>&);
// template<typename cmp>
// bool verify_canonical(graph<cmp>&);

template<typename cmp>
double calc_z_born(graph<cmp>&);
template<typename cmp>
std::vector<double > calc_w_born(graph<cmp>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
template<typename cmp>
std::vector<double> update_cache_w_born(graph<cmp>&,int,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);
std::vector<double> update_cache_w_born(bond&,bond&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&); */

#endif