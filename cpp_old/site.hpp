#ifndef SITE
#define SITE

#include <iostream>
#include <set>
#include <string>

#include "bond.hpp"
#include "bond_utils.cpp"

class site{
public:
    site();
    site(size_t);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const site&);
    size_t vol() const;
    std::multiset<bond,vertices_comparator> adj() const;
    size_t& vol();
    std::multiset<bond,vertices_comparator>& adj();
private:
    size_t vol_;
    std::multiset<bond,vertices_comparator> adj_;
};

#endif
