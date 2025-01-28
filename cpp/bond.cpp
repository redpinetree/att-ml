#include <cmath>
#include <algorithm>

#include "bond.hpp"

bond::bond(){}
bond::bond(int v1,int v2,int depth,array3d<double> w):w_(w),depth_(depth),order_(0),todo_(true){
    this->v_=(v1<v2)?std::pair<int,int>(v1,v2):std::pair<int,int>(v2,v1);
    this->bmi(w);
    this->ee()=this->bmi();
}

bond::bond(std::pair<int,int> v,int depth,array3d<double> w):v_(v),depth_(depth),w_(w),order_(0),todo_(true){
    this->bmi(w);
    this->ee()=this->bmi();
}

bond::operator std::string() const{
    return "[("+std::to_string(this->v1())+","+std::to_string(this->v2())+"),("+std::to_string(this->w().nx())+","+std::to_string(this->w().ny())+","+std::to_string(this->w().nz())+"),"+std::to_string(this->bmi())+","+std::to_string(this->ee())+","+std::to_string(this->depth())+","+std::to_string(this->order())+","+std::to_string(this->todo())+"]";
}

std::ostream& operator<<(std::ostream& os,const bond& e){
    os<<std::string(e);
    return os;
}

std::pair<int,int> bond::v() const{return this->v_;}
int bond::v1() const{return this->v_.first;}
int bond::v2() const{return this->v_.second;}
int bond::depth() const{return this->depth_;}
int bond::order() const{return this->order_;}
double bond::bmi() const{return this->bmi_;}
double bond::ee() const{return this->ee_;}
bool bond::todo() const{return this->todo_;}
array3d<double> bond::w() const{return this->w_;}
std::pair<int,int>& bond::v(){return this->v_;}
int& bond::v1(){return this->v_.first;}
int& bond::v2(){return this->v_.second;}
int& bond::depth(){return this->depth_;}
int& bond::order(){return this->order_;}
bool& bond::todo(){return this->todo_;}
double& bond::bmi(){return this->bmi_;}
double& bond::ee(){return this->ee_;}
array3d<double>& bond::w(){return this->w_;}

void bond::bmi(array3d<double>& w){
    int max_it=100;
    array3d<double> summed_w_ijk(w.nx(),w.ny(),1);
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                summed_w_ijk.at(i,j,0)+=exp(w.at(i,j,k));
            }
        }
    }
    array3d<double> p_ij(w.nx(),w.ny(),1);
    std::vector<double> sum_ax0(w.nx());
    std::vector<double> sum_ax1(w.ny());
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            sum_ax0[i]+=summed_w_ijk.at(i,j,0);
            sum_ax1[j]+=summed_w_ijk.at(i,j,0);
        }
    }
    int nonzero_sum_ax0=0;
    int nonzero_sum_ax1=0;
    for(int i=0;i<w.nx();i++){
        nonzero_sum_ax0+=(sum_ax0[i]!=0)?1:0;
    }
    for(int j=0;j<w.ny();j++){
        nonzero_sum_ax1+=(sum_ax1[j]!=0)?1:0;
    }
    //norm_check bool is true if all elements in sum_ax are 0 (clipped to 0 below 1e-10)
    std::vector<bool> close_ax0(w.nx());
    bool norm_check_ax0=1;
    for(int i=0;i<close_ax0.size();i++){
        //if sum_ax is not already 0 (all elements in row/col are 0)
        close_ax0[i]=((sum_ax0[i]==0)||(fabs(sum_ax0[i]-(1/(double) nonzero_sum_ax0))<1e-10))?1:0;
    }
    norm_check_ax0=std::all_of(close_ax0.begin(),close_ax0.end(),[](bool i){return i;});
    std::vector<bool> close_ax1(w.ny());
    bool norm_check_ax1=1;
    for(int j=0;j<close_ax1.size();j++){
        //if sum_ax is not already 0 (all elements in row/col are 0)
        close_ax1[j]=((sum_ax1[j]==0)||(fabs(sum_ax1[j]-(1/(double) nonzero_sum_ax1))<1e-10))?1:0;
    }
    norm_check_ax1=std::all_of(close_ax1.begin(),close_ax1.end(),[](bool i){return i;});
    // std::cout<<"norm_checks: "<<norm_check_ax0<<","<<norm_check_ax1<<","<<(norm_check_ax0&&norm_check_ax1)<<"\n";
    //normalize w
    if(!(norm_check_ax0&&norm_check_ax1)){
        std::vector<double> x(w.nx(),1/(double) nonzero_sum_ax0);
        std::vector<double> x_old(w.nx(),0);
        std::vector<bool> close_x(w.nx(),0);
        for(int i=0;i<close_x.size();i++){
            close_x[i]=((x[i]==0)||(fabs(x[i]-x_old[i])<1e-10))?1:0;
        }
        int x_it=0;
        while(!std::all_of(close_x.begin(),close_x.end(),[](bool i){return i;})){
            if(x_it>=max_it){break;}
            x_old=x;
            //explicit implementation of x'=(ry/rx)*(W(W^Tx)^-1)^-1
            for(int i=0;i<w.nx();i++){
                double e1=0;
                for(int j=0;j<w.ny();j++){
                    double e2=0;
                    for(int k=0;k<x.size();k++){
                        e2+=summed_w_ijk.at(k,j,0)*x_old[k];
                    }
                    e1+=(e2!=0)?summed_w_ijk.at(i,j,0)*(1/e2):0;
                }
                x[i]=(e1!=0)?((double) nonzero_sum_ax1/(double) nonzero_sum_ax0)*(1/e1):0;
            }
            for(int i=0;i<close_x.size();i++){
                close_x[i]=((x[i]==0)||(fabs(x[i]-x_old[i])<1e-10))?1:0;
            }
            x_it++;
        }
        std::vector<double> y(w.ny(),1/(double) nonzero_sum_ax1);
        std::vector<double> y_old(w.ny());
        std::vector<bool> close_y(w.ny(),0);
        for(int i=0;i<close_y.size();i++){
            close_y[i]=((y[i]==0)||(fabs(y[i]-y_old[i])<1e-10))?1:0;
        }
        int y_it=0;
        while(!std::all_of(close_y.begin(),close_y.end(),[](bool i){return i;})){
            if(y_it>=max_it){break;}
            y_old=y;
            //explicit implementation of y'=(rx/ry)*(W^T(Wy)^-1)^-1
            for(int i=0;i<w.ny();i++){
                double e1=0;
                for(int j=0;j<w.nx();j++){
                    double e2=0;
                    for(int k=0;k<y.size();k++){
                        e2+=summed_w_ijk.at(j,k,0)*y_old[k];
                    }
                    e1+=(e2!=0)?summed_w_ijk.at(j,i,0)*(1/e2):0;
                }
                y[i]=(e1!=0)?((double) nonzero_sum_ax0/(double) nonzero_sum_ax1)*(1/e1):0;
            }
            for(int i=0;i<close_y.size();i++){
                close_y[i]=((y[i]==0)||(fabs(y[i]-y_old[i])<1e-10))?1:0;
            }
            y_it++;
        }
        double sum=0;
        for(int i=0;i<p_ij.nx();i++){
            for(int j=0;j<p_ij.ny();j++){
                p_ij.at(i,j,0)=x[i]*summed_w_ijk.at(i,j,0)*y[j];
                sum+=p_ij.at(i,j,0);
            }
        }
        for(int i=0;i<p_ij.nx();i++){
            for(int j=0;j<p_ij.ny();j++){
                p_ij.at(i,j,0)/=sum;
            }
        }
    }
    else{
        for(int i=0;i<w.nx();i++){
            for(int j=0;j<w.ny();j++){
                p_ij.at(i,j,0)=summed_w_ijk.at(i,j,0);
            }
        }
    }
    std::vector<double> p_i(p_ij.nx());
    std::vector<double> p_j(p_ij.ny());
    for(int i=0;i<p_ij.nx();i++){
        for(int j=0;j<p_ij.ny();j++){
            p_i[i]+=p_ij.at(i,j,0);
            p_j[j]+=p_ij.at(i,j,0);
        }
    }
    
    double S_ij=0;
    double S_i=0;
    double S_j=0;
    for(int i=0;i<p_ij.nx();i++){
        for(int j=0;j<p_ij.ny();j++){
            S_ij-=(p_ij.at(i,j,0)==0)?0:(p_ij.at(i,j,0)*log(p_ij.at(i,j,0)));
        }
    }
    for(int i=0;i<p_ij.nx();i++){
        S_i-=(p_i[i]==0)?0:(p_i[i]*log(p_i[i]));
    }
    for(int j=0;j<p_ij.ny();j++){
        S_j-=(p_j[j]==0)?0:(p_j[j]*log(p_j[j]));
    }
    this->bmi_=S_i+S_j-S_ij;
}

//ver. 1 2025/01/29 - initial
void bond::save(std::ostream& os){
    int ver=1;
    os.write(reinterpret_cast<const char*>(&ver),sizeof(ver)); //version
    os.write(reinterpret_cast<const char*>(&(this->todo())),sizeof(this->todo()));
    os.write(reinterpret_cast<const char*>(&(this->v1())),sizeof(this->v1()));
    os.write(reinterpret_cast<const char*>(&(this->v2())),sizeof(this->v2()));
    os.write(reinterpret_cast<const char*>(&(this->depth())),sizeof(this->depth()));
    os.write(reinterpret_cast<const char*>(&(this->order())),sizeof(this->order()));
    os.write(reinterpret_cast<const char*>(&(this->bmi())),sizeof(this->bmi()));
    os.write(reinterpret_cast<const char*>(&(this->ee())),sizeof(this->ee()));
    this->w().save(os);
}

bond bond::load(std::istream& is){
    bool todo;
    int ver,v1,v2,depth,order;
    double bmi,ee;
    is.read(reinterpret_cast<char*>(&ver),sizeof(ver)); //version
    if(ver!=1){
        std::cout<<"Wrong input version! Expected v1 and got v"<<ver<<".\n";
        exit(1);
    }
    is.read(reinterpret_cast<char*>(&todo),sizeof(todo));
    is.read(reinterpret_cast<char*>(&v1),sizeof(v1));
    is.read(reinterpret_cast<char*>(&v2),sizeof(v2));
    is.read(reinterpret_cast<char*>(&depth),sizeof(depth));
    is.read(reinterpret_cast<char*>(&order),sizeof(order));
    is.read(reinterpret_cast<char*>(&bmi),sizeof(bmi));
    is.read(reinterpret_cast<char*>(&ee),sizeof(ee));
    array3d<double> w=array3d<double>::load(is);
    bond b; //manually load to preserve v order
    b.v1()=v1;
    b.v2()=v2;
    b.todo()=todo;
    b.depth()=depth;
    b.order()=order;
    b.bmi()=bmi;
    b.ee()=ee;
    b.w()=w;
    return b;
}