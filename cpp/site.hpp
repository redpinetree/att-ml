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
    site(size_t,size_t);
    site(size_t,size_t,size_t,size_t,size_t);
    site(size_t,size_t,size_t,std::pair<size_t,size_t>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const site&);
    size_t rank() const;
    size_t vol() const;
    size_t depth() const;
    std::multiset<bond,vertices_comparator> adj() const;
    std::vector<size_t> coords() const;
    bool virt() const;
    std::pair<size_t,size_t> p() const;
    size_t l_idx() const;
    size_t r_idx() const;
    size_t u_idx() const;
    double bmi() const;
    size_t& rank();
    size_t& vol();
    size_t& depth();
    std::multiset<bond,vertices_comparator>& adj();
    std::vector<size_t>& coords();
    bool& virt();
    size_t& l_idx();
    size_t& r_idx();
    size_t& u_idx();
    double& bmi();
    bond& p_bond();
    std::vector<size_t>& orig_ks_idxs();
    std::vector<double>& p_k();
    array3d<double>& p_ijk();
    array2d<double>& p_ik();
    array2d<double>& p_jk();
    std::vector<std::vector<double> >& m_vec();
private:
    size_t rank_;
    size_t vol_;
    size_t depth_;
    std::multiset<bond,vertices_comparator> adj_;
    //only used if a regular lattice is used instead of an arbitrary graph
    std::vector<size_t> coords_;
    //only used in observable computation
    bool virt_;
    size_t l_idx_;
    size_t r_idx_;
    size_t u_idx_;
    double bmi_;
    bond p_bond_;
    std::vector<size_t> orig_ks_idxs_;
    std::vector<double> p_k_;
    array3d<double> p_ijk_;
    array2d<double> p_ik_;
    array2d<double> p_jk_;
    std::vector<std::vector<double> > m_vec_;
};

#endif
