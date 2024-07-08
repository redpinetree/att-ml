#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "algorithm_nll.hpp"
#ifdef MODEL_CMD
#include "cmd/algorithm.hpp"
#endif
#ifdef MODEL_RENYI
#include "renyi/algorithm.hpp"
#endif
#ifdef MODEL_CPD
#include "cpd/algorithm.hpp"
#endif
#include "bond.hpp"
#include "observables.hpp"
#include "optimize_nll.hpp"
#include "sampling.hpp"

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,size_t n_samples,size_t n_sweeps,size_t iter_max){
    double nll=optimize::opt_nll(g,n_samples,n_sweeps,iter_max);
    // double nll=optimize::hopt_nll(g,n_samples,n_sweeps,iter_max);
    // double nll=optimize::hopt_nll2(g,n_samples,n_sweeps,iter_max);
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
}
template void algorithm::train_nll(graph<bmi_comparator>&,size_t,size_t,size_t);

template<typename cmp>
void algorithm::calculate_site_probs(graph<cmp>& g,bond& current){
    double sum=0;
    size_t r_i=g.vs()[current.v1()].rank();
    size_t r_j=g.vs()[current.v2()].rank();
    size_t r_k=g.vs()[current.order()].rank();
    array3d<double> p_ijk(r_i,r_j,r_k);
    array2d<double> p_ik(r_i,r_k);
    array2d<double> p_jk(r_j,r_k);
    std::vector<double> p_k(r_k,0);
    for(size_t i=0;i<r_i;i++){
        for(size_t j=0;j<r_j;j++){
            for(size_t k=0;k<r_k;k++){
                double e=exp(current.w().at(i,j,k));
                p_ijk.at(i,j,k)=e*g.vs()[current.v1()].p_k()[i]*g.vs()[current.v2()].p_k()[j];
                p_ik.at(i,k)+=p_ijk.at(i,j,k); //compute marginals
                p_jk.at(j,k)+=p_ijk.at(i,j,k); //compute marginals
                p_k[k]+=p_ijk.at(i,j,k);
                sum+=p_ijk.at(i,j,k);
            }
        }
    }
    for(size_t k=0;k<p_k.size();k++){
        p_k[k]/=sum;
    }
    double sum_ijk=0;
    for(size_t k=0;k<r_k;k++){
        for(size_t i=0;i<r_i;i++){
            for(size_t j=0;j<r_j;j++){
                sum_ijk+=p_ijk.at(i,j,k);
            }
        }
    }
    for(size_t k=0;k<r_k;k++){
        for(size_t i=0;i<r_i;i++){
            for(size_t j=0;j<r_j;j++){
                p_ijk.at(i,j,k)/=sum_ijk;
            }
        }
    }
    double sum_i=0;
    for(size_t k=0;k<r_k;k++){
        for(size_t i=0;i<r_i;i++){
            sum_i+=p_ik.at(i,k);
        }
    }
    for(size_t k=0;k<r_k;k++){
        for(size_t i=0;i<r_i;i++){
            p_ik.at(i,k)/=sum_i;
        }
    }
    double sum_j=0;
    for(size_t k=0;k<r_k;k++){
        for(size_t j=0;j<r_j;j++){
            sum_j+=p_jk.at(j,k);
        }
    }
    for(size_t k=0;k<r_k;k++){
        for(size_t j=0;j<r_j;j++){
            p_jk.at(j,k)/=sum_j;
        }
    }
    for(size_t k=0;k<r_k;k++){
        for(size_t i=0;i<r_i;i++){
            for(size_t j=0;j<r_j;j++){
                p_ijk.at(i,j,k)/=p_k[k];
            }
        }
    }
    for(size_t k=0;k<r_k;k++){
        for(size_t i=0;i<r_i;i++){
            p_ik.at(i,k)/=p_k[k];
        }
    }
    for(size_t k=0;k<r_k;k++){
        for(size_t j=0;j<r_j;j++){
            p_jk.at(j,k)/=p_k[k];
        }
    }
    g.vs()[current.order()].p_bond()=current;
    g.vs()[current.order()].p_k()=p_k;
    g.vs()[current.order()].p_ijk()=p_ijk;
    g.vs()[current.order()].p_ik()=p_ik;
    g.vs()[current.order()].p_jk()=p_jk;
    
    // for(size_t k=0;k<p_k.size();k++){
        // std::cout<<p_k[k]<<" ";
    // }
    // std::cout<<"\n";
    
    g.vs()[current.order()].m_vec()=std::vector<std::vector<double> >();
    for(size_t idx=0;idx<r_k;idx++){
        std::vector<double> res(r_k-1,0);
        double prob_factor=g.vs()[current.order()].p_k()[idx];
        for(size_t i=0;i<r_k;i++){
            std::vector<double> contrib=observables::m_vec(g,current.order(),i,idx,0);
            for(size_t j=0;j<contrib.size();j++){
                res[j]+=prob_factor*contrib[j];
            }
        }
        g.vs()[current.order()].m_vec().push_back(res);
    }
}
template void algorithm::calculate_site_probs(graph<bmi_comparator>&,bond&);