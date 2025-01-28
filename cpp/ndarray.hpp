#ifndef NDARRAY
#define NDARRAY

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

//kahan sum
template<typename T>
inline double ksum(std::vector<T> v){
    if(v.size()==0){return 0;}
    T c=0; //compensation term
    T sum=v[0];
    for(int i=0;i<v.size()-1;i++){
        T y=v[i+1]-c; //corrected addend
        T t=sum+y; //corrected addend added to current sum
        c=(t-sum)-y; //next compensation
        sum=t;
    }
    return sum;
}

template<typename T>
class array1d{
public:
    array1d():nx_(0){
        this->e_=std::vector<T>();
    }
    array1d(int nx):nx_(nx){
        this->e_=std::vector<T>(nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(int i=0;i<this->nx();i++){
            str<<this->at(i);
            str<<((i==(this->nx()-1))?"]":",");
        }
        if(this->nx()==0){str<<"]";}
        str<<"\n";
        return str.str();
    }
    std::vector<T> sum(){
        ksum(this->e());
    }
    std::vector<T> lse(){
        T res=0;
        std::vector<T> v=this->e();
        double max=*(std::max_element(v.begin(),v.end()));
        for(int n=0;n<v.size();n++){
            res+=exp(v[n]-max);
        }
        res=max+log(e);
        return res;
    }
    //exp for array1d
    array1d<T> exp_form(){
        array1d<double> a=*this;
        for(int i=0;i<a.nx();i++){
            a.at(i)=exp(this->at(i));
        }
        return a;
    }
    int nx() const{return this->nx_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(int x) const{return this->e_[x];}
    T& at(int x){return this->e_[x];}
    
    //ver. 1 2025/01/29 - initial
    void save(std::ostream& os){
        int ver=1;
        os.write(reinterpret_cast<const char*>(&ver),sizeof(ver)); //version
        int nx=this->nx();
        os.write(reinterpret_cast<const char*>(&nx),sizeof(nx));
        for(int i=0;i<this->nx();i++){
            os.write(reinterpret_cast<const char*>(&(this->at(i))),sizeof(this->at(i))); //version
        }
    }

    static array1d<T> load(std::istream& is){
        int ver,nx;
        is.read(reinterpret_cast<char*>(&ver),sizeof(ver)); //version
        if(ver!=1){
            std::cout<<"Wrong input version! Expected v1 and got v"<<ver<<".\n";
            exit(1);
        }
        is.read(reinterpret_cast<char*>(&nx),sizeof(nx));
        array1d<double> w(nx);
        for(int i=0;i<nx;i++){
            T e;
            is.read(reinterpret_cast<char*>(&e),sizeof(e));
            w.at(i)=e;
        }
        return w;
    }
private:
    int nx_;
    std::vector<T> e_;
};

template<typename T>
class array2d{
public:
    array2d():nx_(0),ny_(0){
        this->e_=std::vector<T>(1,0);
    }
    array2d(int nx,int ny):nx_(nx),ny_(ny){
        this->e_=std::vector<T>(ny*nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(int j=0;j<this->ny();j++){
            str<<"[";
            for(int i=0;i<this->nx();i++){
                str<<this->at(i,j);
                str<<((i==(this->nx()-1))?"]":",");
            }
            str<<((j==(this->ny()-1))?"]\n":",\n");
        }
        return str.str();
    }
    std::vector<T> sum_over_axis(int ax){
        if(ax>1){
            std::cerr<<"Attempted to sum over non-existent axis "<<ax<<" in array2d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        if(ax==0){
            for(int j=0;j<this->ny();j++){
                std::vector<T> v;
                for(int i=0;i<this->nx();i++){
                    v.push_back(this->at(i,j));
                }
                res.push_back(ksum(v));
            }
        }
        else if(ax==1){
            for(int i=0;i<this->nx();i++){
                std::vector<T> v;
                for(int j=0;j<this->ny();j++){
                    v.push_back(this->at(i,j));
                }
                res.push_back(ksum(v));
            }
        }
        return res;
    }
    std::vector<T> lse_over_axis(int ax){
        if(ax>1){
            std::cerr<<"Attempted to sum over non-existent axis "<<ax<<" in array2d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        if(ax==0){
            for(int j=0;j<this->ny();j++){
                T e=0;
                std::vector<T> v;
                for(int i=0;i<this->nx();i++){
                    v.push_back(this->at(i,j));
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(int n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        else if(ax==1){
            for(int i=0;i<this->nx();i++){
                T e=0;
                std::vector<T> v;
                for(int j=0;j<this->ny();j++){
                    v.push_back(this->at(i,j));
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(int n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        return res;
    }
    T sum_over_all(){
        return ksum(this->e());
    }
    //exp for array2d
    array2d<T> exp_form(){
        array2d<double> a=*this;
        for(int i=0;i<a.nx();i++){
            for(int j=0;j<a.ny();j++){
                a.at(i,j)=exp(this->at(i,j));
            }
        }
        return a;
    }
    int nx() const{return this->nx_;}
    int ny() const{return this->ny_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(int x,int y) const{return this->e_[(this->nx()*y)+x];}
    T& at(int x,int y){return this->e_[(this->nx()*y)+x];}
    
    //ver. 1 2025/01/29 - initial
    void save(std::ostream& os){
        int ver=1;
        os.write(reinterpret_cast<const char*>(&ver),sizeof(ver)); //version
        int nx=this->nx();
        os.write(reinterpret_cast<const char*>(&nx),sizeof(nx));
        int ny=this->ny();
        os.write(reinterpret_cast<const char*>(&ny),sizeof(ny));
        for(int i=0;i<this->nx();i++){
            for(int j=0;j<this->ny();j++){
                os.write(reinterpret_cast<const char*>(&(this->at(i,j))),sizeof(this->at(i,j))); //version
            }
        }
    }

    static array2d<T> load(std::istream& is){
        int ver,nx,ny;
        is.read(reinterpret_cast<char*>(&ver),sizeof(ver)); //version
        if(ver!=1){
            std::cout<<"Wrong input version! Expected v1 and got v"<<ver<<".\n";
            exit(1);
        }
        is.read(reinterpret_cast<char*>(&nx),sizeof(nx));
        is.read(reinterpret_cast<char*>(&ny),sizeof(ny));
        array2d<double> w(nx,ny);
        for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                T e;
                is.read(reinterpret_cast<char*>(&e),sizeof(e));
                w.at(i,j)=e;
            }
        }
        return w;
    }
private:
    int nx_;
    int ny_;
    std::vector<T> e_;
};

template<typename T>
class array3d{
public:
    array3d():nx_(0),ny_(0),nz_(0){
        this->e_=std::vector<T>(1,0);
    }
    array3d(int nx,int ny,int nz):nx_(nx),ny_(ny),nz_(nz){
        this->e_=std::vector<T>(nz*ny*nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(int k=0;k<this->nz();k++){
            str<<"[";
            for(int j=0;j<this->ny();j++){
                str<<"[";
                for(int i=0;i<this->nx();i++){
                    str<<this->at(i,j,k);
                    str<<((i==(this->nx()-1))?"]":",");
                }
                str<<((j==(this->ny()-1))?"]":",\n");
            }
            str<<((k==(this->nz()-1))?"]\n":",\n\n");
        }
        return str.str();
    }
    std::vector<T> sum_over_axis(int ax0,int ax1){
        if(ax0>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax0<<" in array3d. Aborting...\n";
            exit(1);
        }
        if(ax1>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax1<<" in array3d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        if(((ax0==0)&&(ax1==2))||((ax0==2)&&(ax1==0))){
            for(int j=0;j<this->ny();j++){
                std::vector<T> v;
                for(int i=0;i<this->nx();i++){
                    for(int k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                res.push_back(ksum(v));
            }
        }
        else if(((ax0==1)&&(ax1==2))||((ax0==2)&&(ax1==1))){
            for(int i=0;i<this->nx();i++){
                std::vector<T> v;
                for(int j=0;j<this->ny();j++){
                    for(int k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                res.push_back(ksum(v));
            }
        }
        else if(((ax0==0)&&(ax1==1))||((ax0==1)&&(ax1==0))){
            for(int k=0;k<this->nz();k++){
                std::vector<T> v;
                for(int i=0;i<this->nx();i++){
                    for(int j=0;j<this->ny();j++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                res.push_back(ksum(v));
            }
        }
        return res;
    }
    std::vector<T> lse_over_axis(int ax0,int ax1){
        if(ax0>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax0<<" in array3d. Aborting...\n";
            exit(1);
        }
        if(ax1>2){
            std::cerr<<"Attempted to sum over non-existent axes "<<ax1<<" in array3d. Aborting...\n";
            exit(1);
        }
        std::vector<T> res;
        if(((ax0==0)&&(ax1==2))||((ax0==2)&&(ax1==0))){
            for(int j=0;j<this->ny();j++){
                T e=0;
                std::vector<T> v;
                for(int i=0;i<this->nx();i++){
                    for(int k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(int n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        else if(((ax0==1)&&(ax1==2))||((ax0==2)&&(ax1==1))){
            for(int i=0;i<this->nx();i++){
                T e=0;
                std::vector<T> v;
                for(int j=0;j<this->ny();j++){
                    for(int k=0;k<this->nz();k++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(int n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        else if(((ax0==0)&&(ax1==1))||((ax0==1)&&(ax1==0))){
            for(int k=0;k<this->nz();k++){
                T e=0;
                std::vector<T> v;
                for(int i=0;i<this->nx();i++){
                    for(int j=0;j<this->ny();j++){
                        v.push_back(this->at(i,j,k));
                    }
                }
                std::sort(v.begin(),v.end());
                double max=*(std::max_element(v.begin(),v.end()));
                for(int n=0;n<v.size();n++){
                    e+=exp(v[n]-max);
                }
                e=max+log(e);
                res.push_back(e);
            }
        }
        return res;
    }
    T sum_over_all(){
        return ksum(this->e());
    }
    //exp for array3d
    array3d<T> exp_form(){
        array3d<double> a=*this;
        for(int i=0;i<a.nx();i++){
            for(int j=0;j<a.ny();j++){
                for(int k=0;k<a.nz();k++){
                    a.at(i,j,k)=exp(this->at(i,j,k));
                }
            }
        }
        return a;
    }
    int nx() const{return this->nx_;}
    int ny() const{return this->ny_;}
    int nz() const{return this->nz_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(int x,int y,int z) const{return this->e_[(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
    T& at(int x,int y,int z){return this->e_[(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
    
    //ver. 1 2025/01/29 - initial
    void save(std::ostream& os){
        int ver=1;
        os.write(reinterpret_cast<const char*>(&ver),sizeof(ver)); //version
        int nx=this->nx();
        os.write(reinterpret_cast<const char*>(&nx),sizeof(nx));
        int ny=this->ny();
        os.write(reinterpret_cast<const char*>(&ny),sizeof(ny));
        int nz=this->nz();
        os.write(reinterpret_cast<const char*>(&nz),sizeof(nz));
        for(int i=0;i<this->nx();i++){
            for(int j=0;j<this->ny();j++){
                for(int k=0;k<this->nz();k++){
                    os.write(reinterpret_cast<const char*>(&(this->at(i,j,k))),sizeof(this->at(i,j,k)));
                }
            }
        }
    }

    static array3d<T> load(std::istream& is){
        int ver,nx,ny,nz;
        is.read(reinterpret_cast<char*>(&ver),sizeof(ver)); //version
        if(ver!=1){
            std::cout<<"Wrong input version! Expected v1 and got v"<<ver<<".\n";
            exit(1);
        }
        is.read(reinterpret_cast<char*>(&nx),sizeof(nx));
        is.read(reinterpret_cast<char*>(&ny),sizeof(ny));
        is.read(reinterpret_cast<char*>(&nz),sizeof(nz));
        array3d<double> w(nx,ny,nz);
        for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                for(int k=0;k<nz;k++){
                    T e;
                    is.read(reinterpret_cast<char*>(&e),sizeof(e));
                    w.at(i,j,k)=e;
                }
            }
        }
        return w;
    }
private:
    int nx_;
    int ny_;
    int nz_;
    std::vector<T> e_;
};

template<typename T>
class array4d{
public:
    array4d():nx_(0),ny_(0),nz_(0),nw_(0){
        this->e_=std::vector<T>(1,0);
    }
    array4d(int nx,int ny,int nz,int nw):nx_(nx),ny_(ny),nz_(nz),nw_(nw){
        this->e_=std::vector<T>(nw*nz*ny*nx,0);
    }
    operator std::string() const{
        std::stringstream str;
        str<<"[";
        for(int l=0;l<this->nw();l++){
            str<<"[";
            for(int k=0;k<this->nz();k++){
                str<<"[";
                for(int j=0;j<this->ny();j++){
                    str<<"[";
                    for(int i=0;i<this->nx();i++){
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
    T sum_over_all(){
        return ksum(this->e());
    }
    //exp for array3d
    array4d<T> exp_form(){
        array4d<double> a=*this;
        for(int i=0;i<a.nx();i++){
            for(int j=0;j<a.ny();j++){
                for(int k=0;k<a.nz();k++){
                    for(int l=0;l<a.nw();l++){
                        a.at(i,j,k,l)=exp(this->at(i,j,k,l));
                    }
                }
            }
        }
        return a;
    }
    int nx() const{return this->nx_;}
    int ny() const{return this->ny_;}
    int nz() const{return this->nz_;}
    int nw() const{return this->nw_;}
    std::vector<T> e() const{return this->e_;}
    std::vector<T>& e(){return this->e_;}
    T at(int x,int y,int z,int w) const{return this->e_[(this->nz()*this->ny()*this->nx()*w)+(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
    T& at(int x,int y,int z,int w){return this->e_[(this->nz()*this->ny()*this->nx()*w)+(this->ny()*this->nx()*z)+(this->nx()*y)+x];}
    
    //ver. 1 2025/01/29 - initial
    void save(std::ostream& os){
        int ver=1;
        os.write(reinterpret_cast<const char*>(&ver),sizeof(ver)); //version
        int nx=this->nx();
        os.write(reinterpret_cast<const char*>(&nx),sizeof(nx));
        int ny=this->ny();
        os.write(reinterpret_cast<const char*>(&ny),sizeof(ny));
        int nz=this->nz();
        os.write(reinterpret_cast<const char*>(&nz),sizeof(nz));
        int nw=this->nw();
        os.write(reinterpret_cast<const char*>(&nw),sizeof(nw));
        for(int i=0;i<this->nx();i++){
            for(int j=0;j<this->ny();j++){
                for(int k=0;k<this->nz();k++){
                    for(int l=0;l<this->nw();l++){
                        os.write(reinterpret_cast<const char*>(&(this->at(i,j,k,l))),sizeof(this->at(i,j,k,l))); //version
                    }
                }
            }
        }
    }

    static array4d<T> load(std::istream& is){
        int ver,nx,ny,nz,nw;
        is.read(reinterpret_cast<char*>(&ver),sizeof(ver)); //version
        if(ver!=1){
            std::cout<<"Wrong input version! Expected v1 and got v"<<ver<<".\n";
            exit(1);
        }
        is.read(reinterpret_cast<char*>(&nx),sizeof(nx));
        is.read(reinterpret_cast<char*>(&ny),sizeof(ny));
        is.read(reinterpret_cast<char*>(&nz),sizeof(nz));
        is.read(reinterpret_cast<char*>(&nw),sizeof(nw));
        array4d<double> w(nx,ny,nz,nw);
        for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                for(int k=0;k<nz;k++){
                    for(int l=0;l<nw;l++){
                        T e;
                        is.read(reinterpret_cast<char*>(&e),sizeof(e));
                        w.at(i,j,k,l)=e;
                    }
                }
            }
        }
        return w;
    }
private:
    int nx_;
    int ny_;
    int nz_;
    int nw_;
    std::vector<T> e_;
};

#endif
