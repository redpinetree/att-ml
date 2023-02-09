#include "site.hpp"

site::site(){
    this->vol_=1;
}

site::site(size_t vol){
    this->vol_=vol;
}

site::operator std::string() const{
    return std::to_string(this->vol());
}

std::ostream& operator<<(std::ostream& os,const site& v){
    os<<std::string(v);
    return os;
}

size_t site::vol() const{
    return this->vol_;
}

std::multiset<bond,vertices_comparator> site::adj() const{
    return this->adj_;
}

size_t& site::vol(){
    return this->vol_;
}

std::multiset<bond,vertices_comparator>& site::adj(){
    return this->adj_;
}
