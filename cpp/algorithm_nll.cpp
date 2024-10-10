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

std::vector<sample_data> algorithm::load_training_data_from_file(std::string& fn,size_t& n_samples,size_t& train_data_total_length,size_t& train_data_idim){
    std::vector<sample_data> train_data;
    std::ifstream ifs(fn);
    std::string input_line;
    std::getline(ifs,input_line);
    std::istringstream header(input_line);
    header>>n_samples>>train_data_idim>>train_data_total_length;
    while(std::getline(ifs,input_line)){
        std::istringstream line(input_line);
        std::vector<size_t> s;
        size_t val=0;
        for(size_t i=0;i<train_data_total_length;i++){
            line>>val;
            s.push_back(val+1); //smallest value in sample_data is 1
        }
        train_data.push_back(sample_data(train_data_total_length,s,0,0));
    }
    // for(size_t i=0;i<train_data.size();i++){
        // for(size_t j=0;j<train_data[i].n_phys_sites();j++){
            // std::cout<<train_data[i].s()[j]<<" ";
        // }
        // std::cout<<"\n";
    // }
    // exit(1);
    return train_data;
}

std::vector<size_t> algorithm::load_training_data_labels_from_file(std::string& label_fn,size_t n_samples,size_t& train_data_labels_tdim){
    size_t n_samples_in_file;
    std::vector<size_t> train_data_labels;
    std::ifstream ifs(label_fn);
    std::string input_line;
    std::getline(ifs,input_line);
    std::istringstream header(input_line);
    header>>n_samples_in_file>>train_data_labels_tdim;
    if(n_samples_in_file!=n_samples){
        std::cout<<"Number of samples in the label file does not match the number of samples in the training data file.\n";
        exit(1);
    }
    while(std::getline(ifs,input_line)){
        std::istringstream line(input_line);
        size_t val;
        line>>val;
        train_data_labels.push_back(val);
    }
    // for(size_t i=0;i<train_data_labels.size();i++){
        // std::cout<<train_data_labels[i]<<"\n";
    // }
    // exit(1);
    return train_data_labels;
}

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,size_t n_samples,size_t n_sweeps,size_t iter_max,double lr){
    // std::vector<sample_data> samples=sampling::mh_sample(g,n_samples,0); //consider subtree rooted at n
    // std::vector<sample_data> samples=sampling::local_mh_sample(g,n_samples,n_sweeps,0); //consider subtree rooted at n
    std::vector<sample_data> samples=sampling::hybrid_mh_sample(g,n_samples,n_sweeps,0); //consider subtree rooted at n
    //symmetrize samples
    std::vector<sample_data> sym_samples=sampling::symmetrize_samples(samples);
    std::vector<size_t> dummy_labels;
    double nll=optimize::opt_nll(g,sym_samples,dummy_labels,iter_max,lr);
    // double nll=optimize::hopt_nll(g,n_samples,n_sweeps,iter_max);
    // double nll=optimize::hopt_nll2(g,n_samples,n_sweeps,iter_max);
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
}
template void algorithm::train_nll(graph<bmi_comparator>&,size_t,size_t,size_t,double);

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t iter_max,size_t r_max,bool compress_r,double lr,std::map<size_t,double>& nll_history,bool struct_opt){
    // double nll=optimize::opt_nll(g,samples,labels,iter_max);
    double nll=optimize::opt_struct_nll(g,samples,labels,iter_max,r_max,compress_r,lr,nll_history,struct_opt);
    std::vector<array1d<double> > probs;
    std::vector<size_t> classes=optimize::classify(g,samples,probs);
    double train_acc=0;
    for(size_t s=0;s<labels.size();s++){
        train_acc+=(labels[s]==classes[s]);
    }
    train_acc/=(double) labels.size();
    for(size_t a=0;a<probs.size();a++){
        std::cout<<classes[a]<<" "<<(std::string) probs[a].exp_form();
    }
    std::cout<<"Train acc.="<<train_acc<<"\n";
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w().exp_form()<<"\n";
    // }
}
template void algorithm::train_nll(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,std::map<size_t,double>&,bool);

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,std::vector<sample_data>& samples,size_t iter_max,size_t r_max,bool compress_r,double lr,std::map<size_t,double>& nll_history,bool struct_opt){
    std::vector<size_t> dummy_labels;
    // double nll=optimize::opt_nll(g,samples,dummy_labels,iter_max);
    double nll=optimize::opt_struct_nll(g,samples,dummy_labels,iter_max,r_max,compress_r,lr,nll_history,struct_opt);
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w().exp_form()<<"\n";
    // }
}
template void algorithm::train_nll(graph<bmi_comparator>&,std::vector<sample_data>&,size_t,size_t,bool,double,std::map<size_t,double>&,bool);

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
}
template void algorithm::calculate_site_probs(graph<bmi_comparator>&,bond&);