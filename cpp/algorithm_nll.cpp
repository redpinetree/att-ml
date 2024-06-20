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
std::vector<double> algorithm::train_nll(graph<cmp>& g,size_t n_cycles,size_t n_samples,size_t iter_max){
    std::vector<double> acceptance_ratios;
    double best_acceptance_ratio=0;
    graph<bmi_comparator> best_model=g;
    double acceptance_ratio;
    if(n_cycles==0){
        return acceptance_ratios;
    }
    std::vector<sample_data> samples=sampling::local_mh_sample(g,n_samples);
    std::cout<<n_samples<<" samples generated via local MH\n";
    sampling::mh_sample(g,n_samples,acceptance_ratio);
    // std::vector<sample_data> samples=sampling::mh_sample(g,n_samples,acceptance_ratio);
    acceptance_ratios.push_back(acceptance_ratio);
    // for(size_t s=0;s<n_samples;s+=2){
        // for(size_t m=0;m<g.n_phys_sites();m++){
            // std::cout<<samples[s].s()[m]<<" ";
        // }
        // std::cout<<"\n";
    // }
    for(size_t c=1;c<=n_cycles;c++){
        std::cout<<"cycle "<<c<<"\n";
        double nll=optimize::opt_nll(g,samples,iter_max);
        // double nll=optimize::hopt_nll(g,n_samples,iter_max);
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond current=*it;
            algorithm::calculate_site_probs(g,current);
        }
        samples=sampling::local_mh_sample(g,n_samples);
        std::cout<<n_samples<<" samples generated via local MH\n";
        sampling::mh_sample(g,n_samples,acceptance_ratio);
        // samples=sampling::mh_sample(g,n_samples,acceptance_ratio);
        // acceptance_ratios.push_back(acceptance_ratio);
        if(acceptance_ratio>best_acceptance_ratio){
            best_acceptance_ratio=acceptance_ratio;
            best_model=g;
        }
    }
    g=best_model;
    // for(size_t c=0;c<acceptance_ratios.size();c++){
        // std::cout<<acceptance_ratios[c]<<" ";
    // }
    // std::cout<<"\n";
    return acceptance_ratios;
}

template std::vector<double> algorithm::train_nll(graph<bmi_comparator>&,size_t,size_t,size_t);
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