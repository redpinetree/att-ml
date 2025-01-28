#ifndef SITE
#define SITE

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <utility>

#include "ndarray.hpp"
#include "bond.hpp"
#include "utils.hpp"

class site{
public:
    site();
    site(int,int);
    site(int,int,int,int,int);
    site(int,int,int,std::pair<int,int>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const site&);
    int rank() const;
    int vol() const;
    int depth() const;
    std::vector<int> coords() const;
    bool virt() const;
    std::pair<int,int> p() const;
    int l_idx() const;
    int r_idx() const;
    int u_idx() const;
    double bmi() const;
    double ee() const;
    int& rank();
    int& vol();
    int& depth();
    bool& virt();
    int& l_idx();
    int& r_idx();
    int& u_idx();
    double& bmi();
    double& ee();
    bond& p_bond();
    std::vector<double>& p_k();
    array3d<double>& p_ijk();
    array2d<double>& p_ik();
    array2d<double>& p_jk();
private:
    int rank_;
    int vol_;
    int depth_;
    bool virt_;
    int l_idx_;
    int r_idx_;
    int u_idx_;
    double bmi_;
    double ee_;
    bond p_bond_;
    std::vector<double> p_k_;
    array3d<double> p_ijk_;
    array2d<double> p_ik_;
    array2d<double> p_jk_;
};

#endif
