#ifndef OPTIMIZE_
#define OPTIMIZE_

#include <vector>

#include "site.hpp"
#include "bond.hpp"

namespace optimize{
    void potts_renorm(size_t,std::vector<bond>&,bond&,std::vector<bond>&);
    //debug ver
    void kl_iterative(size_t,size_t,size_t,std::vector<site>,bond&,std::vector<bond>&,bond&,std::vector<bond>&,size_t,double);
    array2d<size_t> f_alg1(site,site);
    array2d<size_t> f_maxent(site,site,array2d<double>,size_t);
    double renorm_coupling(size_t,double,double);
    double kl_div(std::vector<site>,bond,std::vector<bond>,bond,std::vector<bond>);
};

#endif
