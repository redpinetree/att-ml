#ifndef SITE
#define SITE

#include <iostream>
#include <set>
#include <string>
#include <utility>

#include "bond.hpp"
#include "bond_utils.cpp"

class site{
public:
    site();
    site(size_t);
    site(size_t,std::pair<size_t,size_t>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const site&);
    size_t vol() const;
    std::multiset<bond,vertices_comparator> adj() const;
    bool virt() const;
    std::pair<size_t,size_t> p() const;
    size_t p1() const;
    size_t p2() const;
    size_t& vol();
    std::multiset<bond,vertices_comparator>& adj();
    bool& virt();
    std::pair<size_t,size_t>& p();
    size_t& p1();
    size_t& p2();
private:
    size_t vol_;
    std::multiset<bond,vertices_comparator> adj_;
    //only used in observable computation
    bool virt_;
    std::pair<size_t,size_t> p_;
};

#endif
