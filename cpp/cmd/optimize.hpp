#ifndef OPTIMIZE_
#define OPTIMIZE_

#include <vector>

#include "../site.hpp"
#include "bond.hpp"

namespace optimize{
    double opt(size_t,size_t,size_t,std::vector<site>,bond&,std::vector<bond>&,bond&,std::vector<bond>&,size_t,double,size_t);
    array2d<size_t> f_alg1(site,site);
    array2d<size_t> f_maxent(site,site,array3d<double>,size_t);
    array2d<size_t> f_mvec_sim(site,site,array3d<double>,size_t);
    array2d<size_t> f_hybrid_maxent(site,site,array3d<double>,size_t);
    array2d<size_t> f_hybrid_mvec_sim(site,site,array3d<double>,size_t);
    double kl_div(double,double,std::vector<double>,std::vector<double>,bond,std::vector<bond>,bond,std::vector<bond>,array2d<size_t>);
};

#endif
