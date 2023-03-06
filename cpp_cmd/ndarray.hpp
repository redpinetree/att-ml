#ifndef NDARRAY
#define NDARRAY

#include <vector>
#include <sstream>
#include <string>

template<typename T>
class array2d{
public:
    array2d(){
        this->e_=std::vector<T>(1,0);
    }
    array2d(size_t ny,size_t nx):nx_(nx),ny_(ny){
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
        if(ax==0){
            for(size_t j=0;j<this->ny();j++){
                T e=0;
                for(size_t i=0;i<this->nx();i++){
                    e+=this->at(i,j);
                }
                res.push_back(e);
            }
        }
        else if(ax==1){
            for(size_t i=0;i<this->nx();i++){
                T e=0;
                for(size_t j=0;j<this->ny();j++){
                    e+=this->at(i,j);
                }
                res.push_back(e);
            }
        }
        return res;
    }
    T sum_over_all(){
        T res=0;
        for(size_t i=0;i<this->nx();i++){
            for(size_t j=0;j<this->ny();j++){
                res+=this->at(i,j);
            }
        }
        return res;
    }
    size_t nx() const{return this->nx_;}
    size_t ny() const{return this->ny_;}
    T at(size_t y,size_t x) const{return this->e_[(this->nx()*y)+x];}
    T& at(size_t y,size_t x){return this->e_[(this->nx()*y)+x];}
private:
    size_t nx_;
    size_t ny_;
    std::vector<T> e_;
};

template<typename T>
class array3d{
public:
    array3d(){}
    array3d(size_t nz,size_t ny,size_t nx):nx_(nx),ny_(ny),nz_(nz){
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
            str<<((k==(this->ny()-1))?"]\n":",\n\n");
        }
        return str.str();
    }
    size_t nx() const{return this->nx_;}
    size_t ny() const{return this->ny_;}
    size_t nz() const{return this->nz_;}
    T at(size_t z,size_t y,size_t x) const{return this->e_[(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
    T& at(size_t z,size_t y,size_t x){return this->e_[(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
private:
    size_t nx_;
    size_t ny_;
    size_t nz_;
    std::vector<T> e_;
};

#endif
