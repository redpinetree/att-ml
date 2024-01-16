#ifndef GRAPH
#define GRAPH

#include <utility>
#include <vector>
#include <set>
#include <string>

#include "site.hpp"
#include "bond.hpp"

class graph_old{
public:
    graph_old();
    graph_old(std::vector<site>,std::vector<bond>);
    operator std::string() const;
    std::vector<site> vs() const;
    std::vector<bond> es() const;
    std::vector<site>& vs();
    std::vector<bond>& es();
private:
    std::vector<site> vs_;
    std::vector<bond> es_;
};

#endif