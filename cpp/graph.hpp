#ifndef GRAPH
#define GRAPH

#include <vector>
#include <set>
#include <string>

#include "site.hpp"
#include "bond.hpp"

template<typename cmp>
class graph{
public:
    graph();
    graph(std::vector<site>,std::multiset<bond,cmp>);
    operator std::string() const;
    std::vector<site> vs() const;
    std::multiset<bond,cmp> es() const;
    std::vector<site>& vs();
    std::multiset<bond,cmp>& es();
private:
    std::vector<site> vs_;
    std::multiset<bond,cmp> es_;
};

#endif
