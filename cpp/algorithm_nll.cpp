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
#include "optimize_nll_born.hpp"

std::vector<sample_data> algorithm::load_data_from_file(std::string& fn,size_t& n_samples,size_t& data_total_length,size_t& data_idim){
    std::vector<sample_data> data;
    std::ifstream ifs(fn);
    std::string input_line;
    std::getline(ifs,input_line);
    std::istringstream header(input_line);
    header>>n_samples>>data_idim>>data_total_length;
    while(std::getline(ifs,input_line)){
        std::istringstream line(input_line);
        std::vector<size_t> s;
        size_t val=0;
        for(size_t i=0;i<data_total_length;i++){
            line>>val;
            s.push_back(val+1); //smallest value in sample_data is 1
        }
        data.push_back(sample_data(data_total_length,s,0,0));
    }
    // for(size_t i=0;i<data.size();i++){
        // for(size_t j=0;j<data[i].n_phys_sites();j++){
            // std::cout<<data[i].s()[j]<<" ";
        // }
        // std::cout<<"\n";
    // }
    // exit(1);
    return data;
}

std::vector<size_t> algorithm::load_data_labels_from_file(std::string& label_fn,size_t n_samples,size_t& data_labels_tdim){
    size_t n_samples_in_file;
    std::vector<size_t> data_labels;
    std::ifstream ifs(label_fn);
    std::string input_line;
    std::getline(ifs,input_line);
    std::istringstream header(input_line);
    header>>n_samples_in_file>>data_labels_tdim;
    if(n_samples_in_file!=n_samples){
        std::cout<<"Number of samples in the label file does not match the number of samples in the training data file.\n";
        exit(1);
    }
    while(std::getline(ifs,input_line)){
        std::istringstream line(input_line);
        size_t val;
        line>>val;
        data_labels.push_back(val);
    }
    // for(size_t i=0;i<data_labels.size();i++){
        // std::cout<<data_labels[i]<<"\n";
    // }
    // exit(1);
    return data_labels;
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
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
}
template void algorithm::train_nll(graph<bmi_comparator>&,size_t,size_t,size_t,double);

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<size_t>& train_labels,std::vector<sample_data>& test_samples,std::vector<size_t>& test_labels,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,double>& test_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    double nll=optimize::opt_struct_nll(g,train_samples,train_labels,test_samples,test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt);
    std::vector<array1d<double> > train_probs;
    std::vector<size_t> train_classes=optimize::classify(g,train_samples,train_probs);
    double train_acc=0;
    for(size_t s=0;s<train_labels.size();s++){
        train_acc+=(train_labels[s]==train_classes[s]);
    }
    train_acc/=(double) train_labels.size();
    for(size_t a=0;a<train_probs.size();a++){
        std::cout<<train_classes[a]<<" "<<(std::string) train_probs[a].exp_form();
    }
    std::cout<<"Test acc.="<<train_acc<<"\n";
    
    std::vector<array1d<double> > test_probs;
    std::vector<size_t> test_classes=optimize::classify(g,test_samples,test_probs);
    double test_acc=0;
    for(size_t s=0;s<test_labels.size();s++){
        test_acc+=(test_labels[s]==test_classes[s]);
    }
    test_acc/=(double) test_labels.size();
    for(size_t a=0;a<test_probs.size();a++){
        std::cout<<test_classes[a]<<" "<<(std::string) test_probs[a].exp_form();
    }
    std::cout<<"Test acc.="<<test_acc<<"\n";
    
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<size_t>& train_labels,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    std::vector<sample_data> dummy_test_samples;
    std::vector<size_t> dummy_test_labels;
    std::map<size_t,double> dummy_test_nll_history;
    double nll=optimize::opt_struct_nll(g,train_samples,train_labels,dummy_test_samples,dummy_test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,dummy_test_nll_history,sweep_history,struct_opt);
    std::vector<array1d<double> > train_probs;
    std::vector<size_t> train_classes=optimize::classify(g,train_samples,train_probs);
    double train_acc=0;
    for(size_t s=0;s<train_labels.size();s++){
        train_acc+=(train_labels[s]==train_classes[s]);
    }
    train_acc/=(double) train_labels.size();
    for(size_t a=0;a<train_probs.size();a++){
        std::cout<<train_classes[a]<<" "<<(std::string) train_probs[a].exp_form();
    }
    std::cout<<"Train acc.="<<train_acc<<"\n";
    
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<sample_data>& test_samples,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,double>& test_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    std::vector<size_t> dummy_labels;
    std::vector<size_t> dummy_test_labels;
    double nll=optimize::opt_struct_nll(g,train_samples,dummy_labels,test_samples,dummy_test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt);
    
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
void algorithm::train_nll(graph<cmp>& g,std::vector<sample_data>& train_samples,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    std::vector<size_t> dummy_labels;
    std::vector<sample_data> dummy_test_samples;
    std::vector<size_t> dummy_test_labels;
    std::map<size_t,double> dummy_test_nll_history;
    double nll=optimize::opt_struct_nll(g,train_samples,dummy_labels,dummy_test_samples,dummy_test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,dummy_test_nll_history,sweep_history,struct_opt);
    
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll(graph<bmi_comparator>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

#ifdef MODEL_TREE_ML_BORN
template<typename cmp>
void algorithm::train_nll_born(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<size_t>& train_labels,std::vector<sample_data>& test_samples,std::vector<size_t>& test_labels,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,double>& test_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    double nll=optimize::opt_struct_nll_born(g,train_samples,train_labels,test_samples,test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt);
    std::vector<array1d<double> > train_probs;
    std::vector<size_t> train_classes=optimize::classify_born(g,train_samples,train_probs);
    double train_acc=0;
    for(size_t s=0;s<train_labels.size();s++){
        train_acc+=(train_labels[s]==train_classes[s]);
    }
    train_acc/=(double) train_labels.size();
    for(size_t a=0;a<train_probs.size();a++){
        std::cout<<train_classes[a]<<" "<<(std::string) train_probs[a].exp_form();
    }
    std::cout<<"Test acc.="<<train_acc<<"\n";
    
    std::vector<array1d<double> > test_probs;
    std::vector<size_t> test_classes=optimize::classify_born(g,test_samples,test_probs);
    double test_acc=0;
    for(size_t s=0;s<test_labels.size();s++){
        test_acc+=(test_labels[s]==test_classes[s]);
    }
    test_acc/=(double) test_labels.size();
    for(size_t a=0;a<test_probs.size();a++){
        std::cout<<test_classes[a]<<" "<<(std::string) test_probs[a].exp_form();
    }
    std::cout<<"Test acc.="<<test_acc<<"\n";
    
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll_born(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
void algorithm::train_nll_born(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<size_t>& train_labels,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    std::vector<sample_data> dummy_test_samples;
    std::vector<size_t> dummy_test_labels;
    std::map<size_t,double> dummy_test_nll_history;
    double nll=optimize::opt_struct_nll_born(g,train_samples,train_labels,dummy_test_samples,dummy_test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,dummy_test_nll_history,sweep_history,struct_opt);
    std::vector<array1d<double> > train_probs;
    std::vector<size_t> train_classes=optimize::classify_born(g,train_samples,train_probs);
    double train_acc=0;
    for(size_t s=0;s<train_labels.size();s++){
        train_acc+=(train_labels[s]==train_classes[s]);
    }
    train_acc/=(double) train_labels.size();
    for(size_t a=0;a<train_probs.size();a++){
        std::cout<<train_classes[a]<<" "<<(std::string) train_probs[a].exp_form();
    }
    std::cout<<"Train acc.="<<train_acc<<"\n";
    
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll_born(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
void algorithm::train_nll_born(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<sample_data>& test_samples,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,double>& test_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    std::vector<size_t> dummy_labels;
    std::vector<size_t> dummy_test_labels;
    double nll=optimize::opt_struct_nll_born(g,train_samples,dummy_labels,test_samples,dummy_test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt);
    
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll_born(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
void algorithm::train_nll_born(graph<cmp>& g,std::vector<sample_data>& train_samples,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    std::vector<size_t> dummy_labels;
    std::vector<sample_data> dummy_test_samples;
    std::vector<size_t> dummy_test_labels;
    std::map<size_t,double> dummy_test_nll_history;
    double nll=optimize::opt_struct_nll_born(g,train_samples,dummy_labels,dummy_test_samples,dummy_test_labels,iter_max,r_max,compress_r,lr,batch_size,train_nll_history,dummy_test_nll_history,sweep_history,struct_opt);
    
    // std::cout<<(std::string) g<<"\n";
    // for(auto it=g.es().begin();it!=g.es().end();++it){
        // std::cout<<(*it).v1()<<","<<(*it).v2()<<","<<(*it).order()<<"\n";
        // std::cout<<(std::string) (*it).w()<<"\n";
    // }
}
template void algorithm::train_nll_born(graph<bmi_comparator>&,std::vector<sample_data>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);
#endif

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
#if defined(MODEL_TREE_ML) || defined(MODEL_TREE_ML) || defined(MODEL_TREE_ML_BORN)
                double e=current.w().at(i,j,k);
#else
                double e=exp(current.w().at(i,j,k));
#endif
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