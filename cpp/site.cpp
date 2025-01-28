#include "site.hpp"

site::site():rank_(1),vol_(1),depth_(0),virt_(false),bmi_(0){}
site::site(int q,int vol):rank_(q),vol_(vol),depth_(0),virt_(false),bmi_(0){}
site::site(int q,int vol,int depth,int l_idx,int r_idx):rank_(q),vol_(vol),depth_(depth),virt_(true),l_idx_(l_idx),r_idx_(r_idx),bmi_(0){}

site::operator std::string() const{
    if(this->virt()){
        return "["+std::to_string(this->vol())+","+std::to_string(this->rank())+","+std::to_string(this->bmi())+","+std::to_string(this->ee())+","+std::to_string(this->depth())+",("+std::to_string(this->l_idx())+","+std::to_string(this->r_idx())+","+std::to_string(this->u_idx())+")]";
    }
    else{
        return "["+std::to_string(this->vol())+","+std::to_string(this->rank())+","+std::to_string(this->bmi())+","+std::to_string(this->ee())+","+std::to_string(this->depth())+"]";
    }
}

std::ostream& operator<<(std::ostream& os,const site& v){
    os<<std::string(v);
    return os;
}

int site::rank() const{return this->rank_;}
int site::vol() const{return this->vol_;}
int site::depth() const{return this->depth_;}
bool site::virt() const{return this->virt_;}
int site::l_idx() const{return this->l_idx_;}
int site::r_idx() const{return this->r_idx_;}
int site::u_idx() const{return this->u_idx_;}
double site::bmi() const{return this->bmi_;}
double site::ee() const{return this->ee_;}
int& site::rank(){return this->rank_;}
int& site::vol(){return this->vol_;}
int& site::depth(){return this->depth_;}
bool& site::virt(){return this->virt_;}
int& site::l_idx(){return this->l_idx_;}
int& site::r_idx(){return this->r_idx_;}
int& site::u_idx(){return this->u_idx_;}
double& site::bmi(){return this->bmi_;}
double& site::ee(){return this->ee_;}
std::vector<double>& site::p_k(){return this->p_k_;}
array3d<double>& site::p_ijk(){return this->p_ijk_;}
array2d<double>& site::p_ik(){return this->p_ik_;}
array2d<double>& site::p_jk(){return this->p_jk_;}
bond& site::p_bond(){return this->p_bond_;}

//ver. 1 2025/01/29 - initial
void site::save(std::ostream& os){
    int ver=1;
    os.write(reinterpret_cast<const char*>(&ver),sizeof(ver)); //version
    os.write(reinterpret_cast<const char*>(&(this->virt())),sizeof(this->virt()));
    os.write(reinterpret_cast<const char*>(&(this->rank())),sizeof(this->rank()));
    os.write(reinterpret_cast<const char*>(&(this->vol())),sizeof(this->vol()));
    os.write(reinterpret_cast<const char*>(&(this->depth())),sizeof(this->depth()));
    os.write(reinterpret_cast<const char*>(&(this->l_idx())),sizeof(this->l_idx()));
    os.write(reinterpret_cast<const char*>(&(this->r_idx())),sizeof(this->r_idx()));
    os.write(reinterpret_cast<const char*>(&(this->u_idx())),sizeof(this->u_idx()));
    os.write(reinterpret_cast<const char*>(&(this->bmi())),sizeof(this->bmi()));
    os.write(reinterpret_cast<const char*>(&(this->ee())),sizeof(this->ee()));
    int p_k_size=this->p_k().size();
    os.write(reinterpret_cast<const char*>(&p_k_size),sizeof(p_k_size));
    for(int i=0;i<this->p_k().size();i++){
        double e=this->p_k()[i];
        os.write(reinterpret_cast<const char*>(&e),sizeof(e));
    }
    this->p_ijk().save(os);
    this->p_ik().save(os);
    this->p_jk().save(os);
    this->p_bond().save(os);
}

site site::load(std::istream& is){
    bool virt;
    int ver,rank,vol,depth,l_idx,r_idx,u_idx,p_k_size;
    double bmi,ee;
    is.read(reinterpret_cast<char*>(&ver),sizeof(ver)); //version
    if(ver!=1){
        std::cout<<"Wrong input version! Expected v1 and got v"<<ver<<".\n";
        exit(1);
    }
    is.read(reinterpret_cast<char*>(&virt),sizeof(virt));
    is.read(reinterpret_cast<char*>(&rank),sizeof(rank));
    is.read(reinterpret_cast<char*>(&vol),sizeof(vol));
    is.read(reinterpret_cast<char*>(&depth),sizeof(depth));
    is.read(reinterpret_cast<char*>(&l_idx),sizeof(l_idx));
    is.read(reinterpret_cast<char*>(&r_idx),sizeof(r_idx));
    is.read(reinterpret_cast<char*>(&u_idx),sizeof(u_idx));
    is.read(reinterpret_cast<char*>(&bmi),sizeof(bmi));
    is.read(reinterpret_cast<char*>(&ee),sizeof(ee));
    is.read(reinterpret_cast<char*>(&p_k_size),sizeof(p_k_size));
    std::vector<double> p_k(p_k_size);
    for(int i=0;i<p_k_size;i++){
        double e;
        is.read(reinterpret_cast<char*>(&e),sizeof(e));
        p_k[i]=e;
    }
    site s(rank,vol);
    s.virt()=virt;
    s.depth()=depth;
    s.l_idx()=l_idx;
    s.r_idx()=r_idx;
    s.u_idx()=u_idx;
    s.bmi()=bmi;
    s.ee()=ee;
    s.p_k()=p_k;
    s.p_ijk()=array3d<double>::load(is);
    s.p_ik()=array2d<double>::load(is);
    s.p_jk()=array2d<double>::load(is);
    s.p_bond()=bond::load(is);
    return s;
}