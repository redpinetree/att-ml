#include <iostream>
#include <fstream>
#include <queue>
#include <random>
#include <tuple>
#include <vector>

#include "mpi_utils.hpp"
#include "ndarray.hpp"
#include "sampling.hpp"
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
sample_data sampling::sample(size_t root,graph<cmp>& g){
    double log_w=0; //weight of sample (log)
    double e=0; //energy of sample under true hamiltonian
    std::vector<size_t> s(g.vs().size(),0);
    std::queue<size_t> todo_idxs;
    std::discrete_distribution<size_t> pdf(g.vs()[root].p_k().begin(),g.vs()[root].p_k().end());
    s[root]=pdf(mpi_utils::prng)+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
    if(g.vs()[root].virt()){
        todo_idxs.push(root);
    }
    while(!todo_idxs.empty()){
        size_t idx=todo_idxs.front();
        todo_idxs.pop();
        size_t v1=g.vs()[idx].p().first;
        size_t v2=g.vs()[idx].p().second;
        if(g.vs()[v1].virt()){
            todo_idxs.push(v1);
        }
        if(g.vs()[v2].virt()){
            todo_idxs.push(v2);
        }
        std::vector<double> cond_probs;
        // std::cout<<(std::string)g.vs()[idx].p_ijk()<<"\n";
        // for(size_t i=0;i<g.vs()[idx].p_k().size();i++){
            // std::cout<<g.vs()[idx].p_k()[i]<<" ";
        // }
        // std::cout<<"\n";
        for(size_t i=0;i<g.vs()[v1].rank();i++){
            for(size_t j=0;j<g.vs()[v2].rank();j++){
                if(g.vs()[idx].p_k()[s[idx]-1]!=0){
                    cond_probs.push_back(g.vs()[idx].p_ijk().at(i,j,s[idx]-1)); //no need to normalize, handled by discrete_distribution
                }
                else{
                    cond_probs.push_back(0);
                }
            }
        }
        pdf=std::discrete_distribution<size_t>(cond_probs.begin(),cond_probs.end());
        size_t composite_idx=pdf(mpi_utils::prng);
        s[v1]=(composite_idx/g.vs()[v2].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
        s[v2]=(composite_idx%g.vs()[v2].rank())+1; //add 1 to avoid state 0, 0 means empty (to be traced out)
        // log_w+=log(g.vs()[idx].p_ijk().at(s[v1],s[v2],s[idx]));
    }
    for(size_t n=0;n<g.orig_ks().size();n++){
        size_t v1=std::get<0>(g.orig_ks()[n]);
        size_t v2=std::get<1>(g.orig_ks()[n]);
        if((s[v1]!=0)&&(s[v2]!=0)){
            e-=(s[v1]==s[v2])?std::get<2>(g.orig_ks()[n]):0;
        }
    }
    
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
    log_w=lse(w_addends);
    
    // for(size_t m=0;m<s.size();m++){
        // if(m==g.n_phys_sites()){std::cout<<": ";}
        // std::cout<<s[m]<<" ";
    // }
    // std::cout<<exp(log_w)<<" "<<e;
    // std::cout<<"\n";
    return sample_data(g.n_phys_sites(),s,log_w,e);
}
template sample_data sampling::sample(size_t,graph<bmi_comparator>&);


template<typename cmp>
sample_data sampling::sample(graph<cmp>& g){
    sample_data s=sampling::sample(g.vs().size()-1,g);
    return s;
}
template sample_data sampling::sample(graph<bmi_comparator>&);

template<typename cmp>
std::vector<sample_data> sampling::sample(graph<cmp>& g,size_t n_samples){
    std::vector<sample_data> samples;
    samples.reserve(n_samples);
    for(size_t n=0;n<n_samples;n++){
        sample_data s=sampling::sample(g);
        samples.push_back(s);
    }
    return samples;
}
template std::vector<sample_data> sampling::sample(graph<bmi_comparator>&,size_t);

template<typename cmp>
std::vector<sample_data> sampling::mh_sample(graph<cmp>& g,size_t n_samples){
    std::uniform_real_distribution<> unif_dist(0,1.0);
    std::vector<sample_data> samples;
    samples.reserve(n_samples);
    //no need to equilibrate because draws are global and sampling from tree is perfect
    sample_data mc0=sampling::sample(g);
    double accept_ratio=0;
    size_t n=0; //markov chain length
    while(samples.size()<n_samples){
        sample_data mc1=sampling::sample(g);
        double p1=exp(mc0.log_w()-mc1.log_w());
        double p2=exp(-g.beta()*(mc1.e()-mc0.e()));
        double p=p1*p2;
        double r=unif_dist(mpi_utils::prng);
        if(r<p){
            mc0=mc1;
            accept_ratio++;
        }
        //keep every 10th sample, rest are to approximate acceptance ratio
        if((n%100)==0){samples.push_back(mc0);}
        n++;
        // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<(r<p?"accept":"reject")<<"\n";
    }
    accept_ratio/=(double) n;
    std::cout<<"acceptance ratio: "<<accept_ratio<<"\n";
    return samples;
}
template std::vector<sample_data> sampling::mh_sample(graph<bmi_comparator>&,size_t);

template<typename cmp>
std::vector<sample_data> sampling::mh_sample(size_t root,graph<cmp>& g,size_t n_samples){
    std::uniform_real_distribution<> unif_dist(0,1.0);
    std::vector<sample_data> samples;
    samples.reserve(n_samples);
    //no need to equilibrate because draws are global and sampling from tree is perfect
    sample_data mc0=sampling::sample(root,g);
    double accept_ratio=0;
    size_t n=0; //markov chain length
    while(samples.size()<n_samples){
        sample_data mc1=sampling::sample(root,g);
        double p1=exp(mc0.log_w()-mc1.log_w());
        double p2=exp(-g.beta()*(mc1.e()-mc0.e()));
        double p=p1*p2;
        double r=unif_dist(mpi_utils::prng);
        if(r<p){
            mc0=mc1;
            accept_ratio++;
        }
        //keep every 10th sample, rest are to approximate acceptance ratio
        if((n%100)==0){samples.push_back(mc0);}
        n++;
        // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<(r<p?"accept":"reject")<<"\n";
    }
    accept_ratio/=(double) n;
    std::cout<<"acceptance ratio: "<<accept_ratio<<"\n";
    return samples;
}
template std::vector<sample_data> sampling::mh_sample(size_t,graph<bmi_comparator>&,size_t);

template<typename cmp>
std::vector<sample_data> sampling::mh_sample(graph<cmp>& g,size_t n_samples,double& acceptance_ratio){
    std::uniform_real_distribution<> unif_dist(0,1.0);
    std::vector<sample_data> samples;
    samples.reserve(n_samples);
    //no need to equilibrate because draws are global and sampling from tree is perfect
    sample_data mc0=sampling::sample(g);
    acceptance_ratio=0;
    size_t n=0; //markov chain length
    while(samples.size()<n_samples){
        sample_data mc1=sampling::sample(g);
        double p1=exp(mc0.log_w()-mc1.log_w());
        double p2=exp(-g.beta()*(mc1.e()-mc0.e()));
        double p=p1*p2;
        // std::cout<<p1<<" "<<p2<<" "<<sqrt(p2)<<" "<<(p2/(1+p2))<<"\n";
        double r=unif_dist(mpi_utils::prng);
        if(r<p){
            mc0=mc1;
            acceptance_ratio++;
        }
        //keep every 10th sample, rest are to approximate acceptance ratio
        if((n%10)==0){samples.push_back(mc0);}
        n++;
        // std::cout<<p1<<" "<<p2<<" "<<p<<" "<<(r<p?"accept":"reject")<<"\n";
    }
    acceptance_ratio/=(double) n;
    std::cout<<"acceptance ratio: "<<acceptance_ratio<<"\n";
    return samples;
}
template std::vector<sample_data> sampling::mh_sample(graph<bmi_comparator>&,size_t,double&);

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
    //sample means
    double e1_mean=0;
    double e2_mean=0;
    for(size_t n=0;n<samples.size();n++){
        e1_mean+=samples[n].e()/(double) samples[n].n_phys_sites();
        e2_mean+=pow(samples[n].e()/(double) samples[n].n_phys_sites(),2.0);
    }
    e1_mean/=(double) samples.size();
    e2_mean/=(double) samples.size();
    //sample sds
    double e1_sd=0;
    double e2_sd=0;
    for(size_t n=0;n<samples.size();n++){
        e1_sd+=pow((samples[n].e()/(double) samples[n].n_phys_sites())-e1_mean,2.0);
        e2_sd+=pow(pow(samples[n].e()/(double) samples[n].n_phys_sites(),2.0)-e2_mean,2.0);
    }
    e1_sd=sqrt(e1_sd/(double) (samples.size()-1));
    e2_sd=sqrt(e2_sd/(double) (samples.size()-1));
    //prepare output vector
    std::vector<double> res;
    res.push_back(e1_mean);
    res.push_back(e1_sd);
    res.push_back(e2_mean);
    res.push_back(e2_sd);
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
    //sample sds
    double m1_abs_sd=0;
    double m2_sd=0;
    double m4_sd=0;
    for(size_t n=0;n<ms.size();n++){
        m1_abs_sd+=pow(fabs(ms[n])-m1_abs_mean,2.0);
        m2_sd+=pow(pow(ms[n],2.0)-m2_mean,2.0);
        m4_sd+=pow(pow(ms[n],4.0)-m4_mean,4.0);
    }
    m1_abs_sd=sqrt(m1_abs_sd/(double) (ms.size()-1));
    m2_sd=sqrt(m2_sd/(double) (ms.size()-1));
    m4_sd=sqrt(m4_sd/(double) (ms.size()-1));
    //prepare output vector
    std::vector<double> res;
    res.push_back(m1_abs_mean);
    res.push_back(m1_abs_sd);
    res.push_back(m2_mean);
    res.push_back(m2_sd);
    res.push_back(m4_mean);
    res.push_back(m4_sd);
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
    //sample sds
    double q1_abs_sd=0;
    double q2_sd=0;
    double q4_sd=0;
    for(size_t n=0;n<qs.size();n++){
        q1_abs_sd+=pow(fabs(qs[n])-q1_abs_mean,2.0);
        q2_sd+=pow(pow(qs[n],2.0)-q2_mean,2.0);
        q4_sd+=pow(pow(qs[n],4.0)-q4_mean,4.0);
    }
    q1_abs_sd=sqrt(q1_abs_sd/(double) (qs.size()-1));
    q2_sd=sqrt(q2_sd/(double) (qs.size()-1));
    q4_sd=sqrt(q4_sd/(double) (qs.size()-1));
    overlaps=qs; //for output
    //prepare output vector
    std::vector<double> res;
    res.push_back(q1_abs_mean);
    res.push_back(q1_abs_sd);
    res.push_back(q2_mean);
    res.push_back(q2_sd);
    res.push_back(q4_mean);
    res.push_back(q4_sd);
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