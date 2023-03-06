#ifndef GRAPH_
#define GRAPH_

#include <vector>
#include <set>
#include <string>
#include <sstream>

#include "site.hpp"
#include "bond.hpp"

template<typename cmp>
class graph{
public:
    graph(){}
    graph(std::vector<site> vs,std::multiset<bond,cmp> es):vs_(vs),es_(es){};
    operator std::string() const{
        std::ostringstream vs_ss,es_ss;
        for(size_t i=0;i<this->vs().size();i++){
            vs_ss << std::string(this->vs()[i]);
            if(i!=this->vs().size()-1){
                vs_ss << ", ";
            }
        }
        size_t i=0;
        std::multiset<bond,cmp> es=this->es();
        for(auto it=es.begin();it!=es.end();++it){
            es_ss << std::string(*it);
            if(i!=es.size()-1){
                es_ss << ", ";
            }
            i++;
        }
        return "vertices: ["+vs_ss.str()+"]\nedges: ["+es_ss.str()+"]\n";
    }
    std::vector<site> vs() const{return this->vs_;};
    std::multiset<bond,cmp> es() const{return this->es_;};
    std::vector<size_t> dims() const{return this->dims_;};
    std::vector<site>& vs(){return this->vs_;};
    std::multiset<bond,cmp>& es(){return this->es_;};
    std::vector<size_t>& dims(){return this->dims_;};
private:
    std::vector<site> vs_;
    std::multiset<bond,cmp> es_;
    //only nonempty if regular lattice
    std::vector<size_t> dims_;
};

#endif
