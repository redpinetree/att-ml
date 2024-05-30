#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "algorithm_nll.hpp"
#ifdef MODEL_CMD
#include "cmd/algorithm.hpp"
#include "cmd/bond.hpp"
#endif
#ifdef MODEL_RENYI
#include "renyi/algorithm.hpp"
#include "renyi/bond.hpp"
#endif
#ifdef MODEL_CPD
#include "cpd/algorithm.hpp"
#include "cpd/bond.hpp"
#endif
#include "optimize_nll.hpp"
#include "sampling.hpp"

template<typename cmp>
std::vector<double> algorithm::train_nll(graph<cmp>& g,size_t n_cycles,size_t n_samples,size_t iter_max){
    std::vector<double> acceptance_ratios;
    double best_acceptance_ratio=0;
    graph<bmi_comparator> best_model=g;
    double acceptance_ratio;
    std::vector<sample_data> samples=sampling::mh_sample(g,n_samples,acceptance_ratio);
    acceptance_ratios.push_back(acceptance_ratio);
    for(size_t c=1;c<=n_cycles;c++){
        std::cout<<"cycle "<<c<<"\n";
        double nll=optimize::opt_nll(g,samples,iter_max);
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond current=*it;
            algorithm::calculate_site_probs(g,current);
        }
        samples=sampling::mh_sample(g,n_samples,acceptance_ratio);
        acceptance_ratios.push_back(acceptance_ratio);
        if(acceptance_ratio>best_acceptance_ratio){
            best_acceptance_ratio=acceptance_ratio;
            best_model=g;
        }
    }
    g=best_model;
    for(size_t c=0;c<acceptance_ratios.size();c++){
        std::cout<<acceptance_ratios[c]<<" ";
    }
    std::cout<<"\n";
    return acceptance_ratios;
}
template std::vector<double> algorithm::train_nll(graph<bmi_comparator>&,size_t,size_t,size_t);