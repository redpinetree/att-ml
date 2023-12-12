#ifndef UTILS_
#define UTILS_

#include <cmath>
#include <tuple>

#include "bond.hpp"

struct vertices_comparator{
    explicit vertices_comparator(){}
    
    bool operator()(const bond& e1,const bond& e2) const{
        return e1.v()<e2.v();
    }
};

struct bmi_comparator{
    explicit bmi_comparator(){}
    explicit bmi_comparator(size_t q_): q(q_){}
    
    bool operator()(const bond& e1,const bond& e2) const{
        // return std::make_tuple(e1.todo(),e1.bmi(),e1.v())<std::make_tuple(e2.todo(),e2.bmi(),e2.v());
        return std::make_tuple(e1.todo(),e1.order(),e1.bmi(),2-e1.virt_count(),e1.v_orig())<std::make_tuple(e2.todo(),e2.order(),e2.bmi(),2-e2.virt_count(),e2.v_orig());
    }
    
    size_t q;
};

//functions must be inline to avoid repeated compilation
//cartesian product of identical vectors [0,...,q-1], p times
inline std::vector<std::vector<size_t> > spin_cart_prod(size_t q, size_t p){
    std::vector<std::vector<size_t> > cart_prod;
    std::vector<size_t> term(p,0);
    size_t total=1;
    for(size_t i=0;i<p;i++){
        total*=q;
    }
    for(size_t i=0;i<total;i++){
        size_t i_cpy=i;
        for(size_t j=0;j<p;j++){
            term[p-1-j]=i_cpy%q;
            i_cpy/=q;
        }
        cart_prod.push_back(term);
    }
    return cart_prod;
}

//binomial coefficient
inline size_t binom(size_t n,size_t k){
    if(k==0){
        return 1;
    }
    std::vector<size_t> cache(k);
    cache[0]=n-k+1;
    for(size_t i=1;i<k;i++){
        cache[i]=cache[i-1]*(cache[0]+i)/(i+1);
    }
    return cache[k-1];
}

//reference potts vectors
inline std::vector<std::vector<double> > potts_ref_vecs(size_t q){
    std::vector<std::vector<double> > v;
    if(q<2){
        return v;
    }
    for(size_t i=0;i<q;i++){
        v.push_back(std::vector<double>(q-1,0));
    }
    for(size_t i=0;i<(q-1);i++){
        double sum=0;
        for(size_t j=0;j<v[i].size();j++){
            sum+=v[i][j]*v[i][j];
        }
        v[i][i]=sqrt(1-sum);
        for(size_t j=i+1;j<q;j++){
            double dot=0;
            for(size_t k=0;k<v[i].size();k++){
                dot+=v[i][k]*v[j][k];
            }
            v[j][i]=(-(1/(double)(q-1))-dot)/v[i][i];
        }
    }
    // for(size_t i=0;i<v.size();i++){
        // std::cout<<"[";
        // for(size_t j=0;j<v[i].size();j++){
            // std::cout<<v[i][j]<<" ";
        // }
        // std::cout<<"]\n";
    // }
    // std::cout<<"\n";
    return v;
}

#endif