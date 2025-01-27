#include "site.hpp"

site::site():rank_(1),vol_(1),depth_(0),virt_(false),bmi_(0){}
site::site(size_t q,size_t vol):rank_(q),vol_(vol),depth_(0),virt_(false),bmi_(0){}
site::site(size_t q,size_t vol,size_t depth,size_t l_idx,size_t r_idx):rank_(q),vol_(vol),depth_(depth),virt_(true),l_idx_(l_idx),r_idx_(r_idx),bmi_(0){}

site::operator std::string() const{
    if(this->virt()){
        return "["+std::to_string(this->vol())+","+std::to_string(this->rank())+","+std::to_string(this->bmi())+","+std::to_string(this->ee())+","+std::to_string(this->depth())+",("+std::to_string(this->l_idx())+","+std::to_string(this->r_idx())+","+std::to_string(this->u_idx())+")]";
    }
    else{
        return "["+std::to_string(this->vol())+","+std::to_string(this->rank())+","+std::to_string(this->bmi())+","+std::to_string(this->depth())+"]";
    }
}

std::ostream& operator<<(std::ostream& os,const site& v){
    os<<std::string(v);
    return os;
}

size_t site::rank() const{return this->rank_;}
size_t site::vol() const{return this->vol_;}
size_t site::depth() const{return this->depth_;}
bool site::virt() const{return this->virt_;}
size_t site::l_idx() const{return this->l_idx_;}
size_t site::r_idx() const{return this->r_idx_;}
size_t site::u_idx() const{return this->u_idx_;}
double site::bmi() const{return this->bmi_;}
double site::ee() const{return this->ee_;}
size_t& site::rank(){return this->rank_;}
size_t& site::vol(){return this->vol_;}
size_t& site::depth(){return this->depth_;}
bool& site::virt(){return this->virt_;}
size_t& site::l_idx(){return this->l_idx_;}
size_t& site::r_idx(){return this->r_idx_;}
size_t& site::u_idx(){return this->u_idx_;}
double& site::bmi(){return this->bmi_;}
double& site::ee(){return this->ee_;}
bond& site::p_bond(){return this->p_bond_;}
std::vector<double>& site::p_k(){return this->p_k_;}
array3d<double>& site::p_ijk(){return this->p_ijk_;}
array2d<double>& site::p_ik(){return this->p_ik_;}
array2d<double>& site::p_jk(){return this->p_jk_;}
