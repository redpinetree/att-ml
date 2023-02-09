#include <sstream>

#include "graph.hpp"
#include "site.hpp"
#include "bond.hpp"

template<typename cmp>
graph<cmp>::graph(){}
template<typename cmp>
graph<cmp>::graph(std::vector<site> vs,std::multiset<bond,cmp> es):vs_(vs),es_(es){}

template<typename cmp>
graph<cmp>::operator std::string() const{
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

template<typename cmp>
std::vector<site> graph<cmp>::vs() const{return this->vs_;}

template<typename cmp>
std::multiset<bond,cmp> graph<cmp>::es() const{return this->es_;}

template<typename cmp>
std::vector<site>& graph<cmp>::vs(){return this->vs_;}

template<typename cmp>
std::multiset<bond,cmp>& graph<cmp>::es(){return this->es_;}

template class graph<coupling_comparator>;
template class graph<bmi_comparator>;