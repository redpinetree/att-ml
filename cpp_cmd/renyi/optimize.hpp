#ifndef OPTIMIZE_
#define OPTIMIZE_

#include <vector>

#include "../site.hpp"
#include "../bond.hpp"

namespace optimize{
    void opt(size_t,size_t,size_t,std::vector<site>,bond&,std::vector<bond>&,bond&,std::vector<bond>&,size_t,double);
};

#endif
