#include <sstream>

#include "graph.hpp"
#include "site.hpp"
#include "bond.hpp"

graph_old::graph_old(){}
graph_old::graph_old(std::vector<site> vs,std::vector<bond> es): vs_(vs),es_(es){}

graph_old::operator std::string() const{
    std::ostringstream vs_ss,es_ss;
    for(size_t i=0;i<this->vs().size();i++){
        vs_ss << std::string(this->vs()[i]);
        if(i!=this->vs().size()-1){
            vs_ss << ", ";
        }
    }
    for(size_t i=0;i<this->es().size();i++){
        es_ss << std::string(this->es()[i]);
        if(i!=this->es().size()-1){
            es_ss << ", ";
        }
    }
    return "vertices: ["+vs_ss.str()+"]\nedges: ["+es_ss.str()+"]\n";
}

std::vector<site> graph_old::vs() const{return this->vs_;}
std::vector<bond> graph_old::es() const{return this->es_;}
std::vector<site>& graph_old::vs(){return this->vs_;}
std::vector<bond>& graph_old::es(){return this->es_;}