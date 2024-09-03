#ifndef NDARRAY
#define NDARRAY

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

template<typename T>
class array1d{
public:
    array1d():nx_(0){
        this->e_=std::vector<T>();
    }
    array1d(size_t nx):nx_(nx){
        this->e_=std::vector<T>(nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(size_t i=0;i<this->nx();i++){
            str<<this->at(i);
            str<<((i==(this->nx()-1))?"]":",");
        }
        if(this->nx()==0){str<<"]";}
        str<<"\n";
        return str.str();
    }
    std::vector<T> sum(){
        std::vector<T> res;
        //TODO: shewchuk summation
        for(size_t i=0;i<this->nx();i++){
            T e=0;
            // for(size_t j=0;j<this->ny();j++){
                // e+=this->at(i,j);
            // }
            std::vector<T> v;
            for(size_t j=0;j<this->ny();j++){
                v.push_back(this->at(i,j));
            }
            std::sort(v.begin(),v.end());
            for(size_t n=0;n<v.size();n++){
                e+=v[n];
            }
            res.push_back(e);
        }
        return res;
    }
    std::vector<T> lse(){
        std::vector<T> res;
        //TODO: shewchuk summation
        for(size_t i=0;i<this->nx();i++){
            T e=0;
            // for(size_t j=0;j<this->ny();j++){
                // e+=this->at(i,j);
            // }
            std::vector<T> v;
            for(size_t j=0;j<this->ny();j++){
                v.push_back(this->at(i,j));
            }
            std::sort(v.begin(),v.end());
            double max=*(std::max_element(v.begin(),v.end()));
            for(size_t n=0;n<v.size();n++){
                e+=exp(v[n]-max);
            }
            e=max+log(e);
            res.push_back(e);
        }
        return res;
    }
    //exp for array1d
    array1d<T> exp_form(){
        array1d<double> a=*this;
        for(size_t i=0;i<a.nx();i++){
            a.at(i)=exp(this->at(i));
        }
        return a;
    }
    size_t nx() const{return this->nx_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(size_t x) const{return this->e_[x];}
    T& at(size_t x){return this->e_[x];}
private:
    size_t nx_;
    std::vector<T> e_;
};

template<typename T>
class array2d{
public:
    array2d(){
        this->e_=std::vector<T>(1,0);
    }
    array2d(size_t nx,size_t ny):nx_(nx),ny_(ny){
        this->e_=std::vector<T>(ny*nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(size_t j=0;j<this->ny();j++){
            str<<"[";
            for(size_t i=0;i<this->nx();i++){
                str<<this->at(i,j);
                str<<((i==(this->nx()-1))?"]":",");
            }
            str<<((j==(this->ny()-1))?"]\n":",\n");
        }
        return str.str();
    }
    std::vector<T> sum_over_axis(size_t ax){
        if(ax>1){
            std::cerr<<"Attempted to sum over non-existent axis "<<ax<<" in array2d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        //TODO: shewchuk summation
        if(ax==0){
            for(size_t j=0;j<this->ny();j++){
                T e=0;
                // for(size_t i=0;i<this->nx();i++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t i=0;i<this->nx();i++){
                    v.push_back(this->at(i,j));
                }
                std::sort(v.begin(),v.end());
                for(size_t n=0;n<v.size();n++){
                    e+=v[n];
                }
                res.push_back(e);
            }
        }
        else if(ax==1){
            for(size_t i=0;i<this->nx();i++){
                T e=0;
                // for(size_t j=0;j<this->ny();j++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t j=0;j<this->ny();j++){
                    v.push_back(this->at(i,j));
                }
                std::sort(v.begin(),v.end());
                for(size_t n=0;n<v.size();n++){
                    e+=v[n];
                }
                res.push_back(e);
            }
        }
        return res;
    }
    std::vector<T> lse_over_axis(size_t ax){
        if(ax>1){
            std::cerr<<"Attempted to sum over non-existent axis "<<ax<<" in array2d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        //TODO: shewchuk summation
        if(ax==0){
            for(size_t j=0;j<this->ny();j++){
                T e=0;
                // for(size_t i=0;i<this->nx();i++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t i=0;i<this->nx();i++){
                    v.push_back(this->at(i,j));
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(size_t n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        else if(ax==1){
            for(size_t i=0;i<this->nx();i++){
                T e=0;
                // for(size_t j=0;j<this->ny();j++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t j=0;j<this->ny();j++){
                    v.push_back(this->at(i,j));
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(size_t n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        return res;
    }
    T sum_over_all(){ //TODO: sum after sorting
        T res=0;
        for(size_t i=0;i<this->nx();i++){
            for(size_t j=0;j<this->ny();j++){
                res+=this->at(i,j);
            }
        }
        return res;
    }
    //exp for array2d
    array2d<T> exp_form(){
        array2d<double> a=*this;
        for(size_t i=0;i<a.nx();i++){
            for(size_t j=0;j<a.ny();j++){
                a.at(i,j)=exp(this->at(i,j));
            }
        }
        return a;
    }
    size_t nx() const{return this->nx_;}
    size_t ny() const{return this->ny_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(size_t x,size_t y) const{return this->e_[(this->nx()*y)+x];}
    T& at(size_t x,size_t y){return this->e_[(this->nx()*y)+x];}
private:
    size_t nx_;
    size_t ny_;
    std::vector<T> e_;
};

template<typename T>
class array3d{
public:
    array3d(){}
    array3d(size_t nx,size_t ny,size_t nz):nx_(nx),ny_(ny),nz_(nz){
        this->e_=std::vector<T>(nz*ny*nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(size_t k=0;k<this->nz();k++){
            str<<"[";
            for(size_t j=0;j<this->ny();j++){
                str<<"[";
                for(size_t i=0;i<this->nx();i++){
                    str<<this->at(i,j,k);
                    str<<((i==(this->nx()-1))?"]":",");
                }
                str<<((j==(this->ny()-1))?"]":",\n");
            }
            str<<((k==(this->nz()-1))?"]\n":",\n\n");
        }
        return str.str();
    }
    std::vector<T> sum_over_axis(size_t ax0,size_t ax1){
        if(ax0>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax0<<" in array3d. Aborting...\n";
            exit(1);
        }
        if(ax1>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax1<<" in array3d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        //TODO: shewchuk summation
        if(((ax0==0)&&(ax1==2))||((ax0==2)&&(ax1==0))){
            for(size_t j=0;j<this->ny();j++){
                T e=0;
                // for(size_t i=0;i<this->nx();i++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t i=0;i<this->nx();i++){
                    for(size_t k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                for(size_t n=0;n<v.size();n++){
                    e+=v[n];
                }
                res.push_back(e);
            }
        }
        else if(((ax0==1)&&(ax1==2))||((ax0==2)&&(ax1==1))){
            for(size_t i=0;i<this->nx();i++){
                T e=0;
                // for(size_t j=0;j<this->ny();j++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t j=0;j<this->ny();j++){
                    for(size_t k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                for(size_t n=0;n<v.size();n++){
                    e+=v[n];
                }
                res.push_back(e);
            }
        }
        else if(((ax0==0)&&(ax1==1))||((ax0==1)&&(ax1==0))){
            for(size_t k=0;k<this->nz();k++){
                T e=0;
                // for(size_t j=0;j<this->ny();j++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t i=0;i<this->nx();i++){
                    for(size_t j=0;j<this->ny();j++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                for(size_t n=0;n<v.size();n++){
                    e+=v[n];
                }
                res.push_back(e);
            }
        }
        return res;
    }
    std::vector<T> lse_over_axis(size_t ax0,size_t ax1){
        if(ax0>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax0<<" in array3d. Aborting...\n";
            exit(1);
        }
        if(ax1>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax1<<" in array3d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        //TODO: shewchuk summation
        if(((ax0==0)&&(ax1==2))||((ax0==2)&&(ax1==0))){
            for(size_t j=0;j<this->ny();j++){
                T e=0;
                // for(size_t i=0;i<this->nx();i++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t i=0;i<this->nx();i++){
                    for(size_t k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(size_t n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        else if(((ax0==1)&&(ax1==2))||((ax0==2)&&(ax1==1))){
            for(size_t i=0;i<this->nx();i++){
                T e=0;
                // for(size_t j=0;j<this->ny();j++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t j=0;j<this->ny();j++){
                    for(size_t k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(size_t n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        else if(((ax0==0)&&(ax1==1))||((ax0==1)&&(ax1==0))){
            for(size_t k=0;k<this->nz();k++){
                T e=0;
                // for(size_t j=0;j<this->ny();j++){
                    // e+=this->at(i,j);
                // }
                std::vector<T> v;
                for(size_t i=0;i<this->nx();i++){
                    for(size_t j=0;j<this->ny();j++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(size_t n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        return res;
    }
    T sum_over_all(){ //TODO: sum after sorting
        T res=0;
        for(size_t i=0;i<this->nx();i++){
            for(size_t j=0;j<this->ny();j++){
                for(size_t k=0;k<this->nz();k++){
                    res+=this->at(i,j,k);
                }
            }
        }
        return res;
    }
    //exp for array3d
    array3d<T> exp_form(){
        array3d<double> a=*this;
        for(size_t i=0;i<a.nx();i++){
            for(size_t j=0;j<a.ny();j++){
                for(size_t k=0;k<a.nz();k++){
                    a.at(i,j,k)=exp(this->at(i,j,k));
                }
            }
        }
        return a;
    }
    size_t nx() const{return this->nx_;}
    size_t ny() const{return this->ny_;}
    size_t nz() const{return this->nz_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(size_t x,size_t y,size_t z) const{return this->e_[(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
    T& at(size_t x,size_t y,size_t z){return this->e_[(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
private:
    size_t nx_;
    size_t ny_;
    size_t nz_;
    std::vector<T> e_;
};

template<typename T>
class array4d{
public:
    array4d(){}
    array4d(size_t nx,size_t ny,size_t nz,size_t nw):nx_(nx),ny_(ny),nz_(nz),nw_(nw){
        this->e_=std::vector<T>(nw*nz*ny*nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(size_t l=0;l<this->nw();l++){
            str<<"[";
            for(size_t k=0;k<this->nz();k++){
                str<<"[";
                for(size_t j=0;j<this->ny();j++){
                    str<<"[";
                    for(size_t i=0;i<this->nx();i++){
                        str<<this->at(i,j,k,l);
                        str<<((i==(this->nx()-1))?"]":",");
                    }
                    str<<((j==(this->ny()-1))?"]":",\n");
                }
                str<<((k==(this->nz()-1))?"]\n":",\n\n");
            }
            str<<((l==(this->nw()-1))?"]\n\n":",\n\n\n");
        }
        return str.str();
    }
    T sum_over_all(){ //TODO: sum after sorting
        T res=0;
        for(size_t i=0;i<this->nx();i++){
            for(size_t j=0;j<this->ny();j++){
                for(size_t k=0;k<this->nz();k++){
                    for(size_t l=0;l<this->nw();l++){
                        res+=this->at(i,j,k,l);
                    }
                }
            }
        }
        return res;
    }
    //exp for array3d
    array4d<T> exp_form(){
        array4d<double> a=*this;
        for(size_t i=0;i<a.nx();i++){
            for(size_t j=0;j<a.ny();j++){
                for(size_t k=0;k<a.nz();k++){
                    for(size_t l=0;l<a.nw();l++){
                        a.at(i,j,k,l)=exp(this->at(i,j,k,l));
                    }
                }
            }
        }
        return a;
    }
    size_t nx() const{return this->nx_;}
    size_t ny() const{return this->ny_;}
    size_t nz() const{return this->nz_;}
    size_t nw() const{return this->nw_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(size_t x,size_t y,size_t z,size_t w) const{return this->e_[(this->nz()*this->ny()*this->nx()*w)+(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
    T& at(size_t x,size_t y,size_t z,size_t w){return this->e_[(this->nz()*this->ny()*this->nx()*w)+(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
private:
    size_t nx_;
    size_t ny_;
    size_t nz_;
    size_t nw_;
    std::vector<T> e_;
};

#endif
