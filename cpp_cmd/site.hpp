#ifndef SITE
#define SITE

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <utility>

#include "bond.hpp"
#include "bond_utils.cpp"

class site{
public:
    site();
    site(size_t,size_t);
    site(size_t,size_t,size_t,size_t);
    site(size_t,size_t,std::pair<size_t,size_t>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const site&);
    size_t rank() const;
    size_t vol() const;
    std::multiset<bond,vertices_comparator> adj() const;
    std::vector<size_t> coords() const;
    bool virt() const;
    std::pair<size_t,size_t> p() const;
    size_t p1() const;
    size_t p2() const;
    size_t& rank();
    size_t& vol();
    std::multiset<bond,vertices_comparator>& adj();
    std::vector<size_t>& coords();
    bool& virt();
    std::pair<size_t,size_t>& p();
    size_t& p1();
    size_t& p2();
private:
    size_t rank_;
    size_t vol_;
    std::multiset<bond,vertices_comparator> adj_;
    //only used if a regular lattice is used instead of an arbitrary graph
    std::vector<size_t> coords_;
    //only used in observable computation
    bool virt_;
    std::pair<size_t,size_t> p_;
};

#endif
