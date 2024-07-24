#include <iostream>
#include <fstream>
#include <queue>
#include <random>
#include <tuple>
#include <vector>

#include "mpi_utils.hpp"
#include "ndarray.hpp"
#include "sampling.hpp"
#include "ttn_ops.hpp"
#include "utils.hpp"

sample_data::sample_data(){}
sample_data::sample_data(size_t n_phys_sites,std::vector<size_t> s,double log_w, double e):n_phys_sites_(n_phys_sites),s_(s),log_w_(log_w),e_(e){}
size_t sample_data::n_phys_sites() const{return this->n_phys_sites_;}
std::vector<size_t> sample_data::s() const{return this->s_;}
double sample_data::log_w() const{return this->log_w_;}
double sample_data::e() const{return this->e_;}
size_t& sample_data::n_phys_sites(){return this->n_phys_sites_;}
std::vector<size_t>& sample_data::s(){return this->s_;}
double& sample_data::log_w(){return this->log_w_;}
double& sample_data::e(){return this->e_;}

template<typename cmp>
double sampling::calc_sample_e(graph<cmp>& g,std::vector<size_t>& s){
    double e=0; //energy of sample under true hamiltonian
    for(size_t n=0;n<g.orig_ks().size();n++){
        size_t v1=std::get<0>(g.orig_ks()[n]);
        size_t v2=std::get<1>(g.orig_ks()[n]);
        if((s[v1]!=0)&&(s[v2]!=0)){
            e-=(s[v1]==s[v2])?std::get<2>(g.orig_ks()[n]):0;
        }
    }
    return e;
}
template double sampling::calc_sample_e(graph<bmi_comparator>&,std::vector<size_t>&);

template<typename cmp>
double sampling::calc_sample_log_w(graph<cmp>& g,std::vector<size_t>& s){
    //calculate weight of sample tracing out virtual sites
    std::vector<array1d<double> > contracted_vectors;
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites
            array1d<double> vec_e(g.vs()[n].rank());
            if(s[n]!=0){
                for(size_t a=0;a<vec_e.nx();a++){
                    //traced out spins have state 0 and the vector will be all 1s (but in log form)
                    if((s[n]-1)!=a){ //if a==s[n]-1, element is log(1)=0. else log(0)=-inf
                        vec_e.at(a)=log(1e-100);
                    }
                }
            }
            contracted_vectors.push_back(vec_e);
        }
        else{ //virtual sites
            contracted_vectors.push_back(array1d<double>());
        }
    }
    size_t contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()].nx()!=0)&&(contracted_vectors[(*it).v2()].nx()!=0)&&(contracted_vectors[(*it).order()].nx()==0)){ //process if children have been contracted and parent is not yet contracted
                array1d<double> res_vec(g.vs()[(*it).order()].rank());
                for(size_t k=0;k<(*it).w().nz();k++){
                    std::vector<double> res_vec_addends;
                    for(size_t i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(size_t j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                            res_vec_addends.push_back(contracted_vectors[(*it).v1()].at(i)+contracted_vectors[(*it).v2()].at(j)+(*it).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec.at(k)=lse(res_vec_addends); //log space
                }
                contracted_vectors[(*it).order()]=res_vec;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    std::vector<double> w_addends;
    for(size_t k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        w_addends.push_back(contracted_vectors[g.vs().size()-1].at(k));
    }
    double log_w=lse(w_addends);
    return log_w;
}
template double sampling::calc_sample_log_w(graph<bmi_comparator>&,std::vector<size_t>&);

template<typename cmp>
std::vector<sample_data> sampling::random_sample(size_t root,graph<cmp>& g,size_t n_samples){
    std::vector<sample_data> s_vec;
    for(size_t n=0;n<n_samples;n++){
        std::vector<size_t> s(g.vs().size(),0);
        for(size_t e=0;e<s.size();e++){
            s[e]=(mpi_utils::prng()%g.vs()[e].rank())+1;
        }
        double e=calc_sample_e(g,s); //energy of sample under true hamiltonian
        double log_w=-2*g.n_phys_sites(); //weight of sample (log)
        s_vec.push_back(sample_data(g.n_phys_sites(),s,log_w,e));

        // for(size_t m=0;m<s_vec[n].s().size();m++){
            // if(m==g.n_phys_sites()){std::cout<<": ";}
            // std::cout<<s_vec[n].s()[m]<<" ";
        // }
        // std::cout<<exp(log_w)<<" "<<e;
        // std::cout<<"\n";
    }
    return s_vec;
}
template std::vector<sample_data> sampling::random_sample(size_t,graph<bmi_comparator>&,size_t);

template<typename cmp>
std::vector<sample_data> sampling::random_sample(graph<cmp>& g,size_t n_samples){
    std::vector<sample_data> s_vec=sampling::random_sample(g.vs().size()-1,g,n_samples);
    return s_vec;
}
template std::vector<sample_data> sampling::random_sample(graph<bmi_comparator>&,size_t);

template<typename cmp>
std::vector<sample_data> sampling::tree_sample(size_t root,graph<cmp>& g,size_t n_samples){
    std::queue<size_t> todo_idxs;
    std::discrete_distribution<size_t> pdf(g.vs()[root].p_k().begin(),g.vs()[root].p_k().end());
    std::vector<sample_data> s_vec;
    for(size_t n=0;n<n_samples;n++){
        s_vec.push_back(sample_data(g.n_phys_sites(),std::vector<size_t>(g.vs().size(),0),0,0));
        s_vec[n].s()[root]=pdf(mpi_utils::prng)+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
    }
    if(g.vs()[root].virt()){
        todo_idxs.push(root);
    }
    while(!todo_idxs.empty()){
        size_t idx=todo_idxs.front();
        todo_idxs.pop();
        size_t v1=g.vs()[idx].l_idx();
        size_t v2=g.vs()[idx].r_idx();
        if(g.vs()[v1].virt()){
            todo_idxs.push(v1);
        }
        if(g.vs()[v2].virt()){
            todo_idxs.push(v2);
        }
        std::vector<std::discrete_distribution<size_t> > cond_prob_dists;
        // std::cout<<(std::string)g.vs()[idx].p_ijk()<<"\n";
        // for(size_t i=0;i<g.vs()[idx].p_k().size();i++){
            // std::cout<<g.vs()[idx].p_k()[i]<<" ";
        // }
        // std::cout<<"\n";
        for(size_t k=0;k<g.vs()[idx].rank();k++){
            std::vector<double> cond_probs;
            for(size_t i=0;i<g.vs()[v1].rank();i++){
                for(size_t j=0;j<g.vs()[v2].rank();j++){
                    cond_probs.push_back(g.vs()[idx].p_ijk().at(i,j,k)); //no need to normalize, handled by discrete_distribution
                }
            }
            cond_prob_dists.push_back(std::discrete_distribution<size_t>(cond_probs.begin(),cond_probs.end()));
        }
        for(size_t n=0;n<n_samples;n++){
            if(s_vec[n].s()[idx]!=0){
                pdf=cond_prob_dists[s_vec[n].s()[idx]-1];
                size_t composite_idx=pdf(mpi_utils::prng);
                s_vec[n].s()[v1]=(composite_idx/g.vs()[v2].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
                s_vec[n].s()[v2]=(composite_idx%g.vs()[v2].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
            }
        }
    }
    for(size_t n=0;n<n_samples;n++){
        double e=calc_sample_e(g,s_vec[n].s()); //energy of sample under true hamiltonian
        double log_w=calc_sample_log_w(g,s_vec[n].s()); //weight of sample (log)
        s_vec[n].e()=e;
        s_vec[n].log_w()=log_w;
        // for(size_t m=0;m<s_vec[n].s().size();m++){
            // if(m==g.n_phys_sites()){std::cout<<": ";}
            // std::cout<<s_vec[n].s()[m]<<" ";
        // }
        // std::cout<<exp(log_w)<<" "<<e;
        // std::cout<<"\n";
    }
    return s_vec;
}
template std::vector<sample_data> sampling::tree_sample(size_t,graph<bmi_comparator>&,size_t);

template<typename cmp>
std::vector<sample_data> sampling::tree_sample(graph<cmp>& g,size_t n_samples){
    std::vector<sample_data> s_vec=sampling::tree_sample(g.vs().size()-1,g,n_samples);
    return s_vec;
}
template std::vector<sample_data> sampling::tree_sample(graph<bmi_comparator>&,size_t);

template<typename cmp>
std::vector<sample_data> sampling::sample(graph<cmp>& g,size_t n_samples,bool rand_mc){
    std::vector<sample_data> s_vec;
    if(rand_mc){
        s_vec=sampling::random_sample(g,n_samples);
    }
    else{
        s_vec=sampling::tree_sample(g,n_samples);
    }
    return s_vec;
}
template std::vector<sample_data> sampling::sample(graph<bmi_comparator>&,size_t,bool);

template<typename cmp>
std::vector<sample_data> sampling::mh_sample(graph<cmp>& g,size_t n_samples,bool rand_mc){
    double acceptance_ratio; //dummy variable
    std::vector<sample_data> samples=sampling::mh_sample(g.vs().size()-1,g,n_samples,acceptance_ratio,rand_mc);
    return samples;
}
template std::vector<sample_data> sampling::mh_sample(graph<bmi_comparator>&,size_t,bool);

template<typename cmp>
std::vector<sample_data> sampling::mh_sample(graph<cmp>& g,size_t n_samples,double& acceptance_ratio,bool rand_mc){
    std::vector<sample_data> samples=sampling::mh_sample(g.vs().size()-1,g,n_samples,acceptance_ratio,rand_mc);
    std::cout<<"acceptance ratio: "<<acceptance_ratio<<"\n";
    return samples;
}
template std::vector<sample_data> sampling::mh_sample(graph<bmi_comparator>&,size_t,double&,bool);

template<typename cmp>
std::vector<sample_data> sampling::mh_sample(size_t root,graph<cmp>& g,size_t n_samples,double& acceptance_ratio,bool rand_mc){
    std::uniform_real_distribution<> unif_dist(0,1.0);
    std::vector<sample_data> samples;
    //no need to equilibrate because draws are global and sampling from tree is perfect
    sample_data mc0;
    if(rand_mc){
        mc0=sampling::random_sample(root,g,1)[0];
    }
    else{
        mc0=sampling::tree_sample(root,g,1)[0];
    }
    acceptance_ratio=0;
    size_t n=0; //markov chain length
    size_t accepted_count=0; //count of accepted configs, not counting symmetric equivs
    while(accepted_count<n_samples){
        sample_data mc1;
        if(rand_mc){
            mc1=sampling::random_sample(g,1)[0];
        }
        else{
            mc1=sampling::tree_sample(g,1)[0];
        }
        double p1=mc0.log_w()-mc1.log_w();
        double p2=-g.beta()*(mc1.e()-mc0.e());
        double p=p1+p2;
        double r=log(unif_dist(mpi_utils::prng));
        if(r<p){
            mc0=mc1;
            acceptance_ratio++;
        }
        //keep every 100th sample, rest are to approximate acceptance ratio
        if((n%100)==0){
            samples.push_back(mc0);
            accepted_count++;
        }
        n++;
        // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<(r<p?"accept":"reject")<<"\n";
    }
    acceptance_ratio/=(double) n;
    // std::cout<<"acceptance ratio: "<<acceptance_ratio<<"\n";
    return samples;
}
template std::vector<sample_data> sampling::mh_sample(size_t,graph<bmi_comparator>&,size_t,double&,bool);

template<typename cmp>
std::vector<sample_data> sampling::local_mh_sample(graph<cmp>& g,size_t n_samples,size_t n_sweeps,bool rand_mc){
    std::vector<sample_data> samples=sampling::local_mh_sample(g.vs().size()-1,g,n_samples,n_sweeps,rand_mc);
    return samples;
}
template std::vector<sample_data> sampling::local_mh_sample(graph<bmi_comparator>&,size_t,size_t,bool);

template<typename cmp>
std::vector<sample_data> sampling::local_mh_sample(size_t root,graph<cmp>& g,size_t n_samples,size_t n_sweeps,bool rand_mc){
    std::uniform_real_distribution<> unif_dist(0,1.0);
    std::vector<sample_data> samples;
    samples.reserve(n_samples);
    //no need to equilibrate because draws are global and sampling from tree is perfect
    sample_data mc0;
    if(rand_mc){
        mc0=sampling::random_sample(root,g,1)[0];
    }
    else{
        mc0=sampling::tree_sample(root,g,1)[0];
    }
    // double acceptance_ratio=0;
    size_t accepted_count=0; //count of accepted configs, not counting symmetric equivs
    while(accepted_count<n_samples){
        for(size_t sweep=0;sweep<n_sweeps;sweep++){ //number of sweeps
            // for(size_t n=0;n<g.n_phys_sites();n++){
                // std::cout<<mc0.s()[n]<<" ";
            // }
            // std::cout<<"\n";
            for(size_t n=0;n<g.n_phys_sites();n++){
                // std::cout<<"n: "<<n<<"\n";
                size_t new_s;
                double p1,p2;
                // if(rand_mc){
                if(mc0.s()[n]!=0){
                    do{
                        new_s=(mpi_utils::prng()%g.vs()[n].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
                    }
                    while(mc0.s()[n]==new_s);
                    p1=0; //log(1)=0
                }
                // else{
                    // bond current=g.vs()[g.vs()[n].u_idx()].p_bond();
                    // if(mc0.s()[n]!=0){ //ignore traced out sites
                        // std::vector<sample_data> v;
                        // v.push_back(mc0);
                        // std::vector<std::vector<array1d<double> > > l_env;
                        // std::vector<std::vector<array1d<double> > > r_env;
                        // std::vector<std::vector<array1d<double> > > u_env;
                        // calc_w(g,v,l_env,r_env,u_env);
                        // array1d<double> x((n==current.v1())?current.w().nx():current.w().ny()); //calculate weight vector
                        // std::vector<double> sum_addends;
                        // for(size_t i=0;i<current.w().nx();i++){
                            // std::vector<double> addends;
                            // for(size_t j=0;j<current.w().ny();j++){
                                // for(size_t k=0;k<current.w().nz();k++){
                                    // double contrib=current.w().at(i,j,k)+u_env[current.order()][0].at(k)+((n==current.v1())?r_env[current.order()][0].at(j):l_env[current.order()][0].at(i));
                                    // addends.push_back(contrib);
                                    // sum_addends.push_back(contrib);
                                // }
                            // }
                            // x.at(i)=lse(addends);
                        // }
                        // double sum=lse(sum_addends);
                        // for(size_t i=0;i<x.nx();i++){
                            // x.at(i)-=sum;
                        // }
                        // array1d<double> exp_x=x.exp_form();
                        // std::discrete_distribution<size_t> pdf=std::discrete_distribution<size_t>(exp_x.e().begin(),exp_x.e().end());
                        // do{
                            // new_s=pdf(mpi_utils::prng)+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
                        // }
                        // while(mc0.s()[n]==new_s);
                        // p1=x.at(mc0.s()[n])-x.at(new_s);
                    // }
                // }
                double delta_e=0;
                for(size_t m=0;m<g.vs()[n].orig_ks_idxs().size();m++){
                    std::tuple<size_t,size_t,double> k_obj=g.orig_ks()[g.vs()[n].orig_ks_idxs()[m]];
                    size_t v1=std::get<0>(k_obj);
                    size_t v2=std::get<1>(k_obj);
                    // std::cout<<std::get<0>(k_obj)<<" "<<std::get<1>(k_obj)<<" "<<std::get<2>(k_obj) <<"\n";
                    if(v1==n){
                        // std::cout<<g.vs()[n].orig_ks_idxs()[m]<<","<<v1<<","<<v2<<" "<<new_s<<","<<mc0.s()[v2]<<"\n";
                        delta_e+=(new_s!=mc0.s()[v2])?std::get<2>(k_obj):-std::get<2>(k_obj);
                    }
                    else{
                        // std::cout<<g.vs()[n].orig_ks_idxs()[m]<<","<<v1<<","<<v2<<" "<<new_s<<","<<mc0.s()[v1]<<"\n";
                        delta_e+=(new_s!=mc0.s()[v1])?std::get<2>(k_obj):-std::get<2>(k_obj);
                    }
                }
                p2=-g.beta()*delta_e;
                double p=p1+p2;
                double r=log(unif_dist(mpi_utils::prng));
                if(r<p){
                    mc0.s()[n]=new_s;
                }
                // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<r<<" "<<(r<p?"accept":"reject")<<"\n";
            }
        }
        mc0.e()=sampling::calc_sample_e(g,mc0.s());
        samples.push_back(mc0); //always accept draw after some sweeps
        accepted_count++;
    }
    return samples;
}
template std::vector<sample_data> sampling::local_mh_sample(size_t,graph<bmi_comparator>&,size_t,size_t,bool);

template<typename cmp>
std::vector<sample_data> sampling::hybrid_mh_sample(graph<cmp>& g,size_t n_samples,size_t n_sweeps,bool rand_mc){
    std::vector<sample_data> samples=sampling::hybrid_mh_sample(g.vs().size()-1,g,n_samples,n_sweeps,rand_mc);
    return samples;
}
template std::vector<sample_data> sampling::hybrid_mh_sample(graph<bmi_comparator>&,size_t,size_t,bool);

template<typename cmp>
std::vector<std::vector<sample_data> > sampling::hybrid_mh_sample(graph<cmp>& g,size_t n_samples,std::vector<size_t>& n_sweeps_vec,bool rand_mc){
    std::vector<std::vector<sample_data> > samples=sampling::hybrid_mh_sample(g.vs().size()-1,g,n_samples,n_sweeps_vec,rand_mc);
    return samples;
}
template std::vector<std::vector<sample_data> > sampling::hybrid_mh_sample(graph<bmi_comparator>&,size_t,std::vector<size_t>&,bool);

template<typename cmp>
std::vector<sample_data> sampling::hybrid_mh_sample(size_t root,graph<cmp>& g,size_t n_samples,size_t n_sweeps,bool rand_mc){
    std::uniform_real_distribution<> unif_dist(0,1.0);
    //no need to equilibrate because draws are global and sampling from tree is perfect
    std::vector<sample_data> samples;
    if(rand_mc){
        samples=sampling::random_sample(g,n_samples);
    }
    else{
        samples=sampling::tree_sample(g,n_samples);
    }
    for(size_t idx=0;idx<n_samples;idx++){
        for(size_t sweep=0;sweep<n_sweeps;sweep++){ //number of sweeps
            for(size_t n=0;n<g.n_phys_sites();n++){
                // std::cout<<"n: "<<n<<"\n";
                size_t new_s;
                double p1,p2;
                if(samples[idx].s()[n]!=0){
                    do{
                        new_s=(mpi_utils::prng()%g.vs()[n].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
                    }
                    while(samples[idx].s()[n]==new_s);
                    p1=0; //log(1)=0
                }
                double delta_e=0;
                for(size_t m=0;m<g.vs()[n].orig_ks_idxs().size();m++){
                    std::tuple<size_t,size_t,double> k_obj=g.orig_ks()[g.vs()[n].orig_ks_idxs()[m]];
                    size_t v1=std::get<0>(k_obj);
                    size_t v2=std::get<1>(k_obj);
                    if(v1==n){
                        delta_e+=(new_s!=samples[idx].s()[v2])?std::get<2>(k_obj):-std::get<2>(k_obj);
                    }
                    else{
                        delta_e+=(new_s!=samples[idx].s()[v1])?std::get<2>(k_obj):-std::get<2>(k_obj);
                    }
                }
                p2=-g.beta()*delta_e;
                double p=p1+p2;
                double r=log(unif_dist(mpi_utils::prng));
                if(r<p){
                    samples[idx].s()[n]=new_s;
                }
                // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<r<<" "<<(r<p?"accept":"reject")<<"\n";
            }
        }
        // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<(r<p?"accept":"reject")<<"\n";
    }
    return samples;
}
template std::vector<sample_data> sampling::hybrid_mh_sample(size_t,graph<bmi_comparator>&,size_t,size_t,bool);

template<typename cmp>
std::vector<std::vector<sample_data> > sampling::hybrid_mh_sample(size_t root,graph<cmp>& g,size_t n_samples,std::vector<size_t>& n_sweeps_vec,bool rand_mc){
    std::vector<std::vector<sample_data> > res_vec(n_sweeps_vec.size());
    size_t max_n_sweeps=*(std::max_element(n_sweeps_vec.begin(),n_sweeps_vec.end()));
    std::uniform_real_distribution<> unif_dist(0,1.0);
    //no need to equilibrate because draws are global and sampling from tree is perfect
    std::vector<sample_data> samples;
    if(rand_mc){
        samples=sampling::random_sample(g,n_samples);
    }
    else{
        samples=sampling::tree_sample(g,n_samples);
    }
    for(size_t idx=0;idx<n_samples;idx++){
        for(size_t sweep=0;sweep<max_n_sweeps;sweep++){ //number of sweeps
            for(size_t n=0;n<g.n_phys_sites();n++){
                // std::cout<<"n: "<<n<<"\n";
                size_t new_s;
                double p1,p2;
                if(samples[idx].s()[n]!=0){
                    do{
                        new_s=(mpi_utils::prng()%g.vs()[n].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
                    }
                    while(samples[idx].s()[n]==new_s);
                    p1=0; //log(1)=0
                }
                double delta_e=0;
                for(size_t m=0;m<g.vs()[n].orig_ks_idxs().size();m++){
                    std::tuple<size_t,size_t,double> k_obj=g.orig_ks()[g.vs()[n].orig_ks_idxs()[m]];
                    size_t v1=std::get<0>(k_obj);
                    size_t v2=std::get<1>(k_obj);
                    if(v1==n){
                        delta_e+=(new_s!=samples[idx].s()[v2])?std::get<2>(k_obj):-std::get<2>(k_obj);
                    }
                    else{
                        delta_e+=(new_s!=samples[idx].s()[v1])?std::get<2>(k_obj):-std::get<2>(k_obj);
                    }
                }
                p2=-g.beta()*delta_e;
                double p=p1+p2;
                double r=log(unif_dist(mpi_utils::prng));
                if(r<p){
                    samples[idx].s()[n]=new_s;
                }
                // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<r<<" "<<(r<p?"accept":"reject")<<"\n";
            }
            for(size_t sweep_idx=0;sweep_idx<n_sweeps_vec.size();sweep_idx++){
                if((sweep+1)==n_sweeps_vec[sweep_idx]){ //collect intermediate configurations up to max_n_sweeps
                    res_vec[sweep_idx].push_back(samples[idx]);
                    break;
                }
            }
        }
        // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<(r<p?"accept":"reject")<<"\n";
    }
    return res_vec;
}
template std::vector<std::vector<sample_data> > sampling::hybrid_mh_sample(size_t,graph<bmi_comparator>&,size_t,std::vector<size_t>&,bool);

std::vector<sample_data> sampling::symmetrize_samples(std::vector<sample_data>& samples){
    std::vector<sample_data> sym_samples=samples;
    for(size_t s=0;s<samples.size();s++){
        //consider ising symmetry (NEEDS TO BE GENERALIZED)
        sample_data flip=samples[s];
        for(size_t e=0;e<flip.s().size();e++){
            if(flip.s()[e]!=0){
                flip.s()[e]=(flip.s()[e]==1)?2:1; //flip each spin in ising config
            }
        }
        flip.e()=samples[s].e();
        flip.log_w()=samples[s].log_w();
        sym_samples.push_back(flip);
    }
    return sym_samples;
}

template<typename cmp>
void sampling::update_samples(size_t root,graph<cmp>& g,std::vector<sample_data>& samples){
    std::uniform_real_distribution<> unif_dist(0,1.0);
    
    std::vector<array1d<double> > xs;
    std::vector<array1d<double> > exp_xs;
    for(size_t n=0;n<g.n_phys_sites();n++){
        //TODO: unsure where to put this/when to update x
        bond current=g.vs()[g.vs()[n].u_idx()].p_bond();
        std::vector<std::vector<array1d<double> > > l_env;
        std::vector<std::vector<array1d<double> > > r_env;
        std::vector<std::vector<array1d<double> > > u_env;
        calc_w(g,samples,l_env,r_env,u_env);
        array1d<double> x((n==current.v1())?current.w().nx():current.w().ny()); //calculate weight vector
        std::vector<double> sum_addends;
        for(size_t i=0;i<current.w().nx();i++){
            std::vector<double> addends;
            for(size_t j=0;j<current.w().ny();j++){
                for(size_t k=0;k<current.w().nz();k++){
                    double contrib=current.w().at(i,j,k)+u_env[current.order()][0].at(k)+((n==current.v1())?r_env[current.order()][0].at(j):l_env[current.order()][0].at(i));
                    addends.push_back(contrib);
                    sum_addends.push_back(contrib);
                }
            }
            x.at(i)=lse(addends);
        }
        double sum=lse(sum_addends);
        for(size_t i=0;i<x.nx();i++){
            x.at(i)-=sum;
        }
        array1d<double> exp_x=x.exp_form();
        xs.push_back(x);
        exp_xs.push_back(exp_x);
    }
                    
    for(size_t s=0;s<samples.size();s++){
        // std::cout<<"sample "<<s<<"\n";
        for(size_t sweep=0;sweep<10;sweep++){ //number of sweeps
            for(size_t n=0;n<g.n_phys_sites();n++){
                // std::cout<<"n: "<<n<<"\n";
                bond current=g.vs()[g.vs()[n].u_idx()].p_bond();
                bool in_subtree=0;
                size_t upstream_n=n;
                while(g.vs()[upstream_n].depth()<=g.vs()[root].depth()){
                    if(upstream_n==root){
                        in_subtree=1;
                        break;
                    }
                    upstream_n=g.vs()[upstream_n].u_idx();
                }
                // std::cout<<n<<" "<<root<<" "<<in_subtree<<"\n";
                if(in_subtree){
                    std::discrete_distribution<size_t> pdf=std::discrete_distribution<size_t>(exp_xs[n].e().begin(),exp_xs[n].e().end());
                    size_t new_s=pdf(mpi_utils::prng)+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
                    if(samples[s].s()[n]!=new_s){
                        double delta_e=0;
                        for(size_t m=0;m<g.vs()[n].orig_ks_idxs().size();m++){
                            std::tuple<size_t,size_t,double> k_obj=g.orig_ks()[g.vs()[n].orig_ks_idxs()[m]];
                            size_t v1=std::get<0>(k_obj);
                            size_t v2=std::get<1>(k_obj);
                            // std::cout<<g.vs()[n].orig_ks_idxs()[m]<<","<<v1<<","<<v2<<"\n";
                            if(v1==n){
                                delta_e-=(new_s!=samples[s].s()[v2])?std::get<2>(k_obj):0;
                            }
                            else{
                                delta_e-=(new_s!=samples[s].s()[v1])?std::get<2>(k_obj):0;
                            }
                        }
                        double p1=xs[n].at(samples[s].s()[n])-xs[n].at(new_s);
                        double p2=-g.beta()*delta_e;
                        double p=p1+p2;
                        double r=log(unif_dist(mpi_utils::prng));
                        if(r<p){
                            samples[s].s()[n]=new_s;
                        }
                        // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<(r<p?"accept":"reject")<<"\n";
                    }
                }
            }
        }
    }
}
template void sampling::update_samples(size_t,graph<bmi_comparator>&,std::vector<sample_data>&);

//assumes full config
std::vector<double> sampling::pair_overlaps(std::vector<sample_data> samples,size_t q){
    std::vector<double> overlaps;
    for(size_t n1=0;n1<samples.size();n1++){
        for(size_t n2=n1+1;n2<samples.size();n2++){
            double overlap=0;
            for(size_t m=0;m<samples[n1].n_phys_sites();m++){
                overlap+=(samples[n1].s()[m]==samples[n2].s()[m]?1:-1/(double)(q-1));
            }
            overlap/=samples[n1].n_phys_sites();
            overlaps.push_back(overlap);
        }
    }
    // for(size_t m=0;m<overlaps.size();m++){
        // std::cout<<overlaps[m]<<" ";
    // }
    // std::cout<<"\n";
    return overlaps;
}

//assumes full config
std::vector<double> sampling::e_mc(std::vector<sample_data>& samples){
    //sample mean
    double e1_mean=0;
    double e2_mean=0;
    for(size_t n=0;n<samples.size();n++){
        e1_mean+=samples[n].e()/(double) samples[n].n_phys_sites();
        e2_mean+=pow(samples[n].e()/(double) samples[n].n_phys_sites(),2.0);
    }
    e1_mean/=(double) samples.size();
    e2_mean/=(double) samples.size();
    
    //prepare output vector
    std::vector<double> res;
    res.push_back(e1_mean);
    res.push_back(e2_mean);
    return res;
}

//assumes full config
std::vector<double> sampling::m_mc(std::vector<sample_data>& samples,size_t q_orig){
    //determine potts basis vectors
    std::vector<std::vector<double> > ref_basis=potts_ref_vecs(q_orig);
    std::vector<double> ms;
    for(size_t s=0;s<samples.size();s++){
        std::vector<double> vec_m(q_orig-1,0);
        for(size_t e=0;e<samples[s].n_phys_sites();e++){
            std::vector<double> ref_vec=ref_basis[samples[s].s()[e]-1];
            for(size_t r=0;r<vec_m.size();r++){
                vec_m[r]+=ref_vec[r];
            }
        }
        double m=0;
        for(size_t r=0;r<vec_m.size();r++){
            m+=vec_m[r]*vec_m[r];
        }
        m=sqrt(m);
        m/=(double) samples[s].n_phys_sites();
        // std::cout<<m<<"\n";
        ms.push_back(m);
    }
    //sample means
    double m1_abs_mean=0;
    double m2_mean=0;
    double m4_mean=0;
    for(size_t n=0;n<ms.size();n++){
        m1_abs_mean+=fabs(ms[n]);
        m2_mean+=pow(ms[n],2.0);
        m4_mean+=pow(ms[n],4.0);
    }
    m1_abs_mean/=(double) ms.size();
    m2_mean/=(double) ms.size();
    m4_mean/=(double) ms.size();
    
    //prepare output vector
    std::vector<double> res;
    res.push_back(m1_abs_mean);
    res.push_back(m2_mean);
    res.push_back(m4_mean);
    return res;
}

//assumes full config
std::vector<double> sampling::q_mc(std::vector<sample_data>& samples,size_t q_orig,std::vector<double>& overlaps){
    std::vector<double> qs=pair_overlaps(samples,q_orig);
    //sample overlaps
    double q1_abs_mean=0;
    double q2_mean=0;
    double q4_mean=0;
    for(size_t n=0;n<qs.size();n++){
        q1_abs_mean+=fabs(qs[n]);
        q2_mean+=pow(qs[n],2.0);
        q4_mean+=pow(qs[n],4.0);
    }
    q1_abs_mean/=(double) qs.size();
    q2_mean/=(double) qs.size();
    q4_mean/=(double) qs.size();
    overlaps=qs; //for output
    //prepare output vector
    std::vector<double> res;
    res.push_back(q1_abs_mean);
    res.push_back(q2_mean);
    res.push_back(q4_mean);
    return res;
}

double sampling::expected_e(std::vector<sample_data> samples){
    double weighted_e=0;
    double weights_sum=0;
    for(size_t n=0;n<samples.size();n++){
        weighted_e+=exp(samples[n].log_w())*samples[n].e();
        weights_sum+=exp(samples[n].log_w());
    }
    weighted_e/=weights_sum;
    return weighted_e;
}

double sampling::min_e(std::vector<sample_data> samples){
    double min_e=1e50;
    for(size_t n=0;n<samples.size();n++){
        min_e=(samples[n].e()<min_e)?samples[n].e():min_e;
    }
    return min_e;
}