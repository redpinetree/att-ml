#ifndef OPTIMIZE_
#define OPTIMIZE_

#include <map>
#include <utility>
#include <vector>

#include "../site.hpp"
#include "bond.hpp"

namespace optimize{
    array3d<double> calc_aTa(std::vector<array3d<double> >&,std::vector<double>&,size_t);
    array3d<double> calc_aTb(std::vector<array3d<double> >&,bond&,std::vector<bond>&,std::vector<double>&,size_t);
    double calc_tr_axTax(std::vector<array3d<double> >&,std::vector<double>&);
    double calc_tr_axTb(std::vector<array3d<double> >&,bond&,std::vector<bond>&,std::vector<double>&);
    double calc_tr_bTb(std::vector<array3d<double> >&,bond&,std::vector<bond>&);
    array3d<double> nn_hals(array3d<double>&,array3d<double>&,array3d<double>&);
    void normalize(bond&,std::vector<bond>&,std::vector<double>&);
    void unnormalize(bond&,std::vector<double>&);
    double tree_cpd(bond&,std::vector<bond>&,bond&,std::vector<bond>&,size_t,std::string,bool);
    double opt(size_t,size_t,size_t,std::vector<site>,bond&,std::vector<bond>&,bond&,std::vector<bond>&,size_t,std::string,size_t);
};

#endif
