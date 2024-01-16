#include "site.hpp"

site::site():rank_(1),vol_(1),virt_(false){}
site::site(size_t q,size_t vol):rank_(q),vol_(vol),virt_(false){}
site::site(size_t q,size_t vol,size_t p1,size_t p2):rank_(q),vol_(vol),virt_(true),p_(std::make_pair(p1,p2)){}
site::site(size_t q,size_t vol,std::pair<size_t,size_t> p):rank_(q),vol_(vol),virt_(true),p_(p){}

site::operator std::string() const{
    if(this->virt()){
        return "["+std::to_string(this->vol())+",("+std::to_string(this->p1())+","+std::to_string(this->p2())+")]";
    }
    else{
        return std::to_string(this->vol());
    }
}

std::ostream& operator<<(std::ostream& os,const site& v){
    os<<std::string(v);
    return os;
}

size_t site::rank() const{return this->rank_;}
size_t site::vol() const{return this->vol_;}
std::multiset<bond,vertices_comparator> site::adj() const{return this->adj_;}
std::vector<size_t> site::coords() const{return this->coords_;}
bool site::virt() const{return this->virt_;}
std::pair<size_t,size_t> site::p() const{return this->p_;}
size_t site::p1() const{return this->p_.first;}
size_t site::p2() const{return this->p_.second;}
size_t& site::rank(){return this->rank_;}
size_t& site::vol(){return this->vol_;}
std::multiset<bond,vertices_comparator>& site::adj(){return this->adj_;}
std::vector<size_t>& site::coords(){return this->coords_;}
bool& site::virt(){return this->virt_;}
std::pair<size_t,size_t>& site::p(){return this->p_;}
size_t& site::p1(){return this->p_.first;}
size_t& site::p2(){return this->p_.second;}
bond& site::p_bond(){return this->p_bond_;}
std::vector<double>& site::probs(){return this->probs_;}
array3d<double>& site::p_ijk(){return this->p_ijk_;}
array2d<double>& site::p_ik(){return this->p_ik_;}
array2d<double>& site::p_jk(){return this->p_jk_;}
std::vector<std::vector<double> >& site::m_vec(){return this->m_vec_;}
