#include <cmath>
#include <algorithm>
#include <limits>

#include "bond.hpp"

bond::bond(){}
bond::bond(size_t v1,size_t v2,array2d<double> w):w_(w),virt_count_(0),order_(0),todo_(true){
    this->v_=(v1<v2)?std::pair<size_t,size_t>(v1,v2):std::pair<size_t,size_t>(v2,v1);
    this->v_orig_=this->v();
    this->f_=array2d<size_t>(1,1);
    this->j(this->w().nx(),w); //only valid for potts models with constant rank
    this->bmi(w);
}

bond::bond(std::pair<size_t,size_t> v,array2d<double> w):v_(v),v_orig_(v),virt_count_(0),w_(w),order_(0),todo_(true){
    this->f_=array2d<size_t>(1,1);
    this->j(this->w().nx(),w); //only valid for potts models with constant rank
    this->bmi(w);
}

bond::operator std::string() const{
    // return "[("+std::to_string(this->v1())+","+std::to_string(this->v2())+"),"+std::to_string(this->j())+","+std::to_string(this->bmi())+","+std::to_string(this->order())+","+std::to_string(this->todo())+"]";
    return "[("+std::to_string(this->v1())+","+std::to_string(this->v2())+"),("+std::to_string(this->w().nx())+","+std::to_string(this->w().ny())+"),"+std::to_string(this->bmi())+","+std::to_string(this->order())+","+std::to_string(this->todo())+"]";
}

std::ostream& operator<<(std::ostream& os,const bond& e){
    os<<std::string(e);
    return os;
}

// size_t bond::q() const{return this->q_;}
std::pair<size_t,size_t> bond::v() const{return this->v_;}
std::pair<size_t,size_t> bond::v_orig() const{return this->v_orig_;}
size_t bond::v1() const{return this->v_.first;}
size_t bond::v2() const{return this->v_.second;}
size_t bond::v1_orig() const{return this->v_orig_.first;}
size_t bond::v2_orig() const{return this->v_orig_.second;}
size_t bond::virt_count() const{return this->virt_count_;}
array2d<double> bond::w() const{return this->w_;}
array2d<size_t> bond::f() const{return this->f_;}
double bond::j() const{return this->j_;}
double bond::bmi() const{return this->bmi_;}
size_t bond::order() const{return this->order_;}
bool bond::todo() const{return this->todo_;}
// size_t& bond::q(){return this->q_;}
std::pair<size_t,size_t>& bond::v(){return this->v_;}
std::pair<size_t,size_t>& bond::v_orig(){return this->v_orig_;}
size_t& bond::v1(){return this->v_.first;}
size_t& bond::v2(){return this->v_.second;}
size_t& bond::v1_orig(){return this->v_orig_.first;}
size_t& bond::v2_orig(){return this->v_orig_.second;}
size_t& bond::virt_count(){return this->virt_count_;}
double& bond::j(){return this->j_;}
array2d<double>& bond::w(){return this->w_;}
array2d<size_t>& bond::f(){return this->f_;}
double& bond::bmi(){return this->bmi_;}
size_t& bond::order(){return this->order_;}
bool& bond::todo(){return this->todo_;}

//only sensible for potts models
void bond::j(size_t q,array2d<double>& w){
    double arg=((1/w.at(0,0))-w.nx())/(double) (w.nx()*(w.nx()-1));
    // this->j_=(arg>0)?-log(arg):-INFINITY;
    this->j_=-log(arg);
}

void bond::bmi(array2d<double>& w){
    // if(this->j()==-INFINITY){
        // this->bmi_=0;
        // return;
    // }
    array2d<double> p_ij(w.nx(),w.ny());
    std::vector<double> sum_ax0(w.nx());
    std::vector<double> sum_ax1(w.ny());
    for(size_t i=0;i<w.nx();i++){
        for(size_t j=0;j<w.ny();j++){
            sum_ax0[i]+=w.at(i,j);
            sum_ax1[j]+=w.at(i,j);
        }
    }
    size_t nonzero_sum_ax0=0;
    size_t nonzero_sum_ax1=0;
    for(size_t i=0;i<w.nx();i++){
        nonzero_sum_ax0+=(sum_ax0[i]!=0)?1:0;
    }
    for(size_t j=0;j<w.ny();j++){
        nonzero_sum_ax1+=(sum_ax1[j]!=0)?1:0;
    }
    // std::cout<<"pre bmi:\n"<<(std::string) w<<"\n";
        // std::vector<double> row_sum=w.sum_over_axis(0);
        // std::vector<double> col_sum=w.sum_over_axis(1);
        // for(size_t j=0;j<col_sum.size();j++){
            // std::cout<<col_sum[j]<<",";
        // }
        // std::cout<<"\n";
        // for(size_t i=0;i<row_sum.size();i++){
            // std::cout<<row_sum[i]<<",";
        // }
        // std::cout<<"\n\n";
    //norm_check bool is true if all elements in sum_ax are 0 (clipped to 0 below 1e-10)
    std::vector<bool> close_ax0(w.nx());
    bool norm_check_ax0=1;
    for(size_t i=0;i<close_ax0.size();i++){
        //if sum_ax is not already 0 (all elements in row/col are 0)
        close_ax0[i]=((sum_ax0[i]==0)||(fabs(sum_ax0[i]-(1/(double) nonzero_sum_ax0))<1e-10))?1:0;
    }
    // for(size_t i=0;i<close_ax0.size();i++){
        // std::cout<<close_ax0[i]<<" ";
    // }
    // std::cout<<"\n";
    norm_check_ax0=std::all_of(close_ax0.begin(),close_ax0.end(),[](bool i){return i;});
    std::vector<bool> close_ax1(w.ny());
    bool norm_check_ax1=1;
    for(size_t j=0;j<close_ax1.size();j++){
        //if sum_ax is not already 0 (all elements in row/col are 0)
        close_ax1[j]=((sum_ax1[j]==0)||(fabs(sum_ax1[j]-(1/(double) nonzero_sum_ax1))<1e-10))?1:0;
    }
    // for(size_t j=0;j<close_ax1.size();j++){
        // std::cout<<close_ax1[j]<<" ";
    // }
    // std::cout<<"\n";
    norm_check_ax1=std::all_of(close_ax1.begin(),close_ax1.end(),[](bool i){return i;});
    // std::cout<<"norm_checks: "<<norm_check_ax0<<","<<norm_check_ax1<<","<<(norm_check_ax0&&norm_check_ax1)<<"\n";
    //normalize w
    if(!(norm_check_ax0&&norm_check_ax1)){
        // std::cout<<"x:\n";
        std::vector<double> x(w.nx(),1/(double) nonzero_sum_ax0);
        std::vector<double> x_old(w.nx());
        // for(size_t i=0;i<x.size();i++){
            // std::cout<<x[i]<<" ";
        // }
        // std::cout<<"\n";
        std::vector<bool> close_x(w.nx(),0);
        for(size_t i=0;i<close_x.size();i++){
            close_x[i]=((x[i]==0)||(fabs(x[i]-x_old[i])<1e-10))?1:0;
        }
        // std::cout<<"check: "<<std::all_of(close_x.begin(),close_x.end(),[](bool i){return i;})<<"\n";
        while(!std::all_of(close_x.begin(),close_x.end(),[](bool i){return i;})){
            x_old=x;
            //explicit implementation of x'=(ry/rx)*(W(W^Tx)^-1)^-1
            for(size_t i=0;i<w.nx();i++){
                double e1=0;
                for(size_t j=0;j<w.ny();j++){
                    double e2=0;
                    for(size_t k=0;k<x.size();k++){
                        e2+=w.at(k,j)*x_old[k];
                    }
                    e1+=(e2!=0)?w.at(i,j)*(1/e2):0;
                }
                x[i]=(e1!=0)?((double) nonzero_sum_ax1/(double) nonzero_sum_ax0)*(1/e1):0;
            }
            for(size_t i=0;i<close_x.size();i++){
                close_x[i]=((x[i]==0)||(fabs(x[i]-x_old[i])<1e-10))?1:0;
            }
            // for(size_t i=0;i<close_x.size();i++){
                // std::cout<<x[i]<<" ";
            // }
            // std::cout<<"\n";
        }
        // std::cout<<"y:\n";
        std::vector<double> y(w.ny(),1/(double) nonzero_sum_ax1);
        std::vector<double> y_old(w.ny());
        // for(size_t i=0;i<y.size();i++){
            // std::cout<<y[i]<<" ";
        // }
        // std::cout<<"\n";
        std::vector<bool> close_y(w.ny(),0);
        for(size_t i=0;i<close_y.size();i++){
            close_y[i]=((y[i]==0)||(fabs(y[i]-y_old[i])<1e-10))?1:0;
        }
        while(!std::all_of(close_y.begin(),close_y.end(),[](bool i){return i;})){
            y_old=y;
            //explicit implementation of y'=(rx/ry)*(W^T(Wy)^-1)^-1
            for(size_t i=0;i<w.ny();i++){
                double e1=0;
                for(size_t j=0;j<w.nx();j++){
                    double e2=0;
                    for(size_t k=0;k<y.size();k++){
                        e2+=w.at(j,k)*y_old[k];
                    }
                    e1+=(e2!=0)?w.at(j,i)*(1/e2):0;
                }
                y[i]=(e1!=0)?((double) nonzero_sum_ax0/(double) nonzero_sum_ax1)*(1/e1):0;
            }
            for(size_t i=0;i<close_y.size();i++){
                close_y[i]=((y[i]==0)||(fabs(y[i]-y_old[i])<1e-10))?1:0;
            }
            // for(size_t i=0;i<close_y.size();i++){
                // std::cout<<y[i]<<" ";
            // }
            // std::cout<<"\n";
        }
        double sum=0;
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                p_ij.at(i,j)=x[i]*w.at(i,j)*y[j];
                sum+=p_ij.at(i,j);
            }
        }
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                p_ij.at(i,j)/=sum;
            }
        }
        w=p_ij; //needed?
    }
    else{
        p_ij=w;
    }
    // std::cout<<"post bmi:\n"<<(std::string) w<<"\n";
        // row_sum=w.sum_over_axis(0);
        // col_sum=w.sum_over_axis(1);
        // for(size_t j=0;j<col_sum.size();j++){
            // std::cout<<col_sum[j]<<",";
        // }
        // std::cout<<"\n";
        // for(size_t i=0;i<row_sum.size();i++){
            // std::cout<<row_sum[i]<<",";
        // }
        // std::cout<<"\n\n";
    std::vector<double> p_i(p_ij.nx());
    std::vector<double> p_j(p_ij.ny());
    for(size_t i=0;i<p_ij.nx();i++){
        for(size_t j=0;j<p_ij.ny();j++){
            p_i[i]+=p_ij.at(i,j);
            p_j[j]+=p_ij.at(i,j);
        }
    }
    // for(size_t i=0;i<p_i.size();i++){
        // std::cout<<p_i[i]<<" ";
    // }
    // std::cout<<"\n";
    // for(size_t j=0;j<p_j.size();j++){
        // std::cout<<p_j[j]<<" ";
    // }
    // std::cout<<"\n";
    double S_ij=0;
    double S_i=0;
    double S_j=0;
    for(size_t i=0;i<p_ij.nx();i++){
        for(size_t j=0;j<p_ij.ny();j++){
            S_ij-=(p_ij.at(i,j)==0)?0:(p_ij.at(i,j)*log(p_ij.at(i,j)));
        }
    }
    for(size_t i=0;i<p_ij.nx();i++){
        S_i-=(p_i[i]==0)?0:(p_i[i]*log(p_i[i]));
    }
    for(size_t j=0;j<p_ij.ny();j++){
        S_j-=(p_j[j]==0)?0:(p_j[j]*log(p_j[j]));
    }
    this->bmi_=S_i+S_j-S_ij;
}