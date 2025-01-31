#ifndef UTILS_
#define UTILS_

#include <cmath>
#include <tuple>

#include "bond.hpp"

struct vertices_comparator{
    explicit vertices_comparator(){}
    
    inline bool operator()(const bond& e1,const bond& e2) const{
        return e1.v()<e2.v();
    }
};

struct bmi_comparator{
    explicit bmi_comparator(){}
    explicit bmi_comparator(int q_): q(q_){}
    
    inline bool operator()(const bond& e1,const bond& e2) const{
        // return std::make_tuple(e1.todo(),e1.bmi(),e1.v())<std::make_tuple(e2.todo(),e2.bmi(),e2.v());
        return std::make_tuple(e1.todo(),e1.depth(),e1.order(),e1.bmi(),e1.v())<std::make_tuple(e2.todo(),e2.depth(),e2.order(),e2.bmi(),e2.v());
    }
    
    int q;
};

//functions must be inline to avoid repeated compilation
//logsumexp
inline double lse(double a,double b){
    return (a>b)?a+log(1+exp(b-a)):b+log(exp(a-b)+1);
}
inline double lse(std::vector<double> v){
    std::sort(v.begin(),v.end());
    if(v.size()==0){
        return 0;
    }
    double max=*(std::max_element(v.begin(),v.end()));
    double sum=0;
    for(int i=0;i<v.size();i++){
        sum+=exp(v[i]-max);
    }
    sum=max+log(sum);
    return sum;
}

//sorted float add
// inline double vec_add_float(std::vector<double> v){
    // std::sort(v.begin(),v.end());
    // double sum=0;
    // for(int i=0;i<v.size();i++){
        // sum+=v[i];
    // }
    // return sum;
// }

//kahan sum
inline double vec_add_float(std::vector<double> v){
    if(v.size()==0){return 0;}
    double c=0; //compensation term
    double sum=v[0];
    for(int i=0;i<v.size()-1;i++){
        double y=v[i+1]-c; //corrected addend
        double t=sum+y; //corrected addend added to current sum
        c=(t-sum)-y; //next compensation
        sum=t;
    }
    return sum;
}

//sorted float mult
inline double vec_mult_float(std::vector<double> v){
    std::sort(v.begin(),v.end());
    double prod=1;
    for(int i=0;i<v.size();i++){
        prod*=v[i];
    }
    return prod;
}

//cartesian product of identical vectors [0,...,q-1], p times
inline std::vector<std::vector<int> > spin_cart_prod(int q, int p){
    std::vector<std::vector<int> > cart_prod;
    std::vector<int> term(p,0);
    int total=1;
    for(int i=0;i<p;i++){
        total*=q;
    }
    for(int i=0;i<total;i++){
        int i_cpy=i;
        for(int j=0;j<p;j++){
            term[p-1-j]=i_cpy%q;
            i_cpy/=q;
        }
        cart_prod.push_back(term);
    }
    return cart_prod;
}

//cartesian product of vectors
inline std::vector<std::vector<int> > spin_cart_prod(std::vector<int> v){
    std::vector<std::vector<int> > cart_prod;
    std::vector<int> term(v.size(),0);
    int total=1;
    for(int i=0;i<v.size();i++){
        total*=v[i];
    }
    for(int i=0;i<total;i++){
        int i_cpy=i;
        for(int j=0;j<v.size();j++){
            term[v.size()-1-j]=i_cpy%v[j];
            i_cpy/=v[j];
        }
        cart_prod.push_back(term);
    }
    return cart_prod;
}

//binomial coefficient
inline int binom(int n,int k){
    if(k==0){
        return 1;
    }
    std::vector<int> cache(k);
    cache[0]=n-k+1;
    for(int i=1;i<k;i++){
        cache[i]=cache[i-1]*(cache[0]+i)/(i+1);
    }
    return cache[k-1];
}

//reference potts vectors
inline std::vector<std::vector<double> > potts_ref_vecs(int q){
    std::vector<std::vector<double> > v;
    if(q<2){
        return v;
    }
    for(int i=0;i<q;i++){
        v.push_back(std::vector<double>(q-1,0));
    }
    for(int i=0;i<(q-1);i++){
        double sum=0;
        for(int j=0;j<v[i].size();j++){
            sum+=v[i][j]*v[i][j];
        }
        v[i][i]=sqrt(1-sum);
        for(int j=i+1;j<q;j++){
            double dot=0;
            for(int k=0;k<v[i].size();k++){
                dot+=v[i][k]*v[j][k];
            }
            v[j][i]=(-(1/(double)(q-1))-dot)/v[i][i];
        }
    }
    // for(int i=0;i<v.size();i++){
        // std::cout<<"[";
        // for(int j=0;j<v[i].size();j++){
            // std::cout<<v[i][j]<<" ";
        // }
        // std::cout<<"]\n";
    // }
    // std::cout<<"\n";
    return v;
}
#endif