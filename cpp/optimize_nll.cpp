#include <fstream>

#include "mpi_utils.hpp"
#include "optimize_nll.hpp"
#include "observables.hpp"

template<typename cmp>
double optimize::opt_nll(graph<cmp>& g,std::vector<sample_data> samples,size_t iter_max){
    double prev_nll=1e50;
    double nll=0;
    //adam variables
    double alpha=0.01;
    double beta1=0.9;
    double beta2=0.999;
    double epsilon=1e-8;
    //initialize adam m,v caches
    std::vector<array3d<double> > m;
    std::vector<array3d<double> > v;
    std::multiset<bond,bmi_comparator> new_es;
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        m.push_back(array3d<double>(current.w().nx(),current.w().ny(),current.w().nz()));
        v.push_back(array3d<double>(current.w().nx(),current.w().ny(),current.w().nz()));
    }
    
    for(size_t t=1;t<=iter_max;t++){
        size_t n_samples=samples.size();
        std::vector<array1d<double> > l_env_z;
        std::vector<array1d<double> > r_env_z;
        std::vector<array1d<double> > u_env_z;
        double z=optimize::calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
        std::vector<array3d<double> > dz=optimize::calc_dz(l_env_z,r_env_z,u_env_z); //index i corresponds to tensor with order i so some (for input sites) are empty
        
        std::vector<std::vector<array1d<double> > > l_env_sample;
        std::vector<std::vector<array1d<double> > > r_env_sample;
        std::vector<std::vector<array1d<double> > > u_env_sample;
        std::vector<double> w=optimize::calc_w(g,samples,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        std::vector<std::vector<array3d<double> > > dw=optimize::calc_dw(l_env_sample,r_env_sample,u_env_sample); //index i corresponds to tensor with order i so some (for input sites) are empty
        
        size_t n=0;
        std::multiset<bond,bmi_comparator> new_es;
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond current=*it;
            array3d<double> grad(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_z_term(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_w_term(current.w().nx(),current.w().ny(),current.w().nz());
            // std::cout<<"n="<<(n+g.n_phys_sites())<<"\n";
            // std::cout<<(std::string) current.w()<<"\n";
            std::vector<double> sum_addends;
            for(size_t i=0;i<grad_z_term.nx();i++){
                for(size_t j=0;j<grad_z_term.ny();j++){
                    for(size_t k=0;k<grad_z_term.nz();k++){
                        grad_z_term.at(i,j,k)=dz[n+g.n_phys_sites()].at(i,j,k)-z; //log space
                        std::vector<double> grad_w_term_addends;
                        for(size_t s=0;s<n_samples;s++){
                            grad_w_term_addends.push_back(dw[n+g.n_phys_sites()][s].at(i,j,k)-w[s]); //log space
                        }
                        // std::cout<<lse(grad_w_term_addends)<<" "<<log(n_samples)<<"\n";
                        grad_w_term.at(i,j,k)=lse(grad_w_term_addends)-log(n_samples);
                        // grad.at(i,j,k)=(exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k)))*exp(current.w().at(i,j,k));
                        // current.w().at(i,j,k)-=0.1*grad.at(i,j,k);
                        grad.at(i,j,k)=(exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k)))*exp(current.w().at(i,j,k));
                        m[n].at(i,j,k)=(beta1*m[n].at(i,j,k))+((1-beta1)*grad.at(i,j,k));
                        v[n].at(i,j,k)=(beta2*v[n].at(i,j,k))+((1-beta2)*grad.at(i,j,k)*grad.at(i,j,k));
                        double corrected_m=m[n].at(i,j,k)/(1-pow(beta1,(double) t));
                        double corrected_v=v[n].at(i,j,k)/(1-pow(beta2,(double) t));
                        current.w().at(i,j,k)-=alpha*(corrected_m/(sqrt(corrected_v)+epsilon));
                        // std::cout<<current.w().at(i,j,k)<<"\n";
                        sum_addends.push_back(current.w().at(i,j,k));
                    }
                }
            }
            double sum=lse(sum_addends);
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<current.w().nz();k++){
                        current.w().at(i,j,k)-=sum;
                        if(current.w().at(i,j,k)<log(1e-100)){current.w().at(i,j,k)=log(1e-100);}
                    }
                }
            }
            // std::cout<<(std::string) grad_z_term.exp_form()<<"\n";
            // std::cout<<(std::string) grad_w_term.exp_form()<<"\n";
            // std::cout<<(std::string) grad<<"\n";
            new_es.insert(current);
            n++;
        }
        g.es()=new_es;
        // std::cout<<"new\n";
        // for(auto it=g.es().begin();it!=g.es().end();++it){
            // std::cout<<(std::string) (*it).w()<<"\n";
        // }
        //calculate nll and check for convergence
        nll=0;
        for(size_t s=0;s<n_samples;s++){
            nll-=w[s]; //w[s] is log(w(s))
        }
        nll/=(double) n_samples;
        nll+=z; //z is log(z)
        if(fabs(prev_nll-nll)<1e-8){
            std::cout<<"NLL optimization converged after "<<t<<" iterations.\n";
            std::cout<<"nll="<<nll<<"\n";
            break;
        }
        else{
            if(((t-1)%100)==0){
                std::cout<<"nll="<<nll<<"\n";
            }
        }
        prev_nll=nll;
    }
    return nll;
}
template double optimize::opt_nll(graph<bmi_comparator>&,std::vector<sample_data>,size_t);

std::vector<array3d<double> > optimize::calc_dz(std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){ //calculate it for all tensors simultaneously
    std::vector<array3d<double> > res;
    for(size_t n=0;n<u_env.size();n++){
        array3d<double> res_element(l_env[n].nx(),r_env[n].nx(),u_env[n].nx());
        for(size_t i=0;i<l_env[n].nx();i++){
            for(size_t j=0;j<r_env[n].nx();j++){
                for(size_t k=0;k<u_env[n].nx();k++){
                    res_element.at(i,j,k)=l_env[n].at(i)+r_env[n].at(j)+u_env[n].at(k); //log space
                }
            }
        }
        res.push_back(res_element);
    }
    return res;
}

template<typename cmp>
double optimize::calc_z(graph<cmp>& g){
    size_t contracted_idx_count=0;
    std::vector<array1d<double> > contracted_vectors;
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            contracted_vectors.push_back(array1d<double>(g.vs()[n].rank())); //0 because log(1)=0
            contracted_idx_count++;
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(array1d<double>());
        }
    }
    while(contracted_idx_count!=g.vs().size()){
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()].nx()!=0)&&(contracted_vectors[(*it).v2()].nx()!=0)&&(contracted_vectors[(*it).order()].nx()==0)){ //process if children have been contracted and parent is not yet contracted (check if size is 1)
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
                if(contracted_idx_count==g.vs().size()){break;}
            }
        }
    }
    std::vector<double> z_addends;
    for(size_t k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z_addends.push_back(contracted_vectors[g.vs().size()-1].at(k));
    }
    double z=lse(z_addends);
    return z;
}
template double optimize::calc_z(graph<bmi_comparator>&);

template<typename cmp>
double optimize::calc_z(graph<cmp>& g,std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){
    std::vector<array1d<double> > contracted_vectors;
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            contracted_vectors.push_back(array1d<double>(g.vs()[n].rank())); //0 because log(1)=0
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(array1d<double>());
        }
        l_env.push_back(array1d<double>());
        r_env.push_back(array1d<double>());
        u_env.push_back(array1d<double>());
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
                l_env[(*it).order()]=contracted_vectors[(*it).v1()];
                r_env[(*it).order()]=contracted_vectors[(*it).v2()];
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            if((u_env[(*it).v1()].nx()==0)&&(u_env[(*it).v2()].nx()==0)){
                if(contracted_idx_count==0){ //top tensor's u_env is all ones
                    u_env[(*it).order()]=array1d<double>(g.vs()[(*it).order()].rank()); //0 because log(1)=0
                }
                array1d<double> res_vec_l(g.vs()[(*it).v1()].rank());
                for(size_t i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                    std::vector<double> res_vec_l_addends;
                    for(size_t j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            res_vec_l_addends.push_back(r_env[(*it).order()].at(j)+u_env[(*it).order()].at(k)+(*it).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec_l.at(i)=lse(res_vec_l_addends); //log space
                }
                u_env[(*it).v1()]=res_vec_l;
                array1d<double> res_vec_r(g.vs()[(*it).v2()].rank());
                for(size_t j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                    std::vector<double> res_vec_r_addends;
                    for(size_t i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            res_vec_r_addends.push_back(l_env[(*it).order()].at(i)+u_env[(*it).order()].at(k)+(*it).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec_r.at(j)=lse(res_vec_r_addends); //log space
                }
                u_env[(*it).v2()]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    std::vector<double> z_addends;
    for(size_t k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z_addends.push_back(contracted_vectors[g.vs().size()-1].at(k));
    }
    double z=lse(z_addends);
    return z;
}
template double optimize::calc_z(graph<bmi_comparator>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);

std::vector<std::vector<array3d<double> > > optimize::calc_dw(std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){ //calculate it for all tensors simultaneously
    std::vector<std::vector<array3d<double> > > res;
    for(size_t n=0;n<u_env.size();n++){
        std::vector<array3d<double> > batched_res_vec;
        for(size_t s=0;s<u_env[0].size();s++){
            array3d<double> res_element(l_env[n][s].nx(),r_env[n][s].nx(),u_env[n][s].nx());
            for(size_t i=0;i<l_env[n][s].nx();i++){
                for(size_t j=0;j<r_env[n][s].nx();j++){
                    for(size_t k=0;k<u_env[n][s].nx();k++){
                        res_element.at(i,j,k)=l_env[n][s].at(i)+r_env[n][s].at(j)+u_env[n][s].at(k); //log space
                    }
                }
            }
            batched_res_vec.push_back(res_element);
        }
        res.push_back(batched_res_vec);
    }
    return res;
}

template<typename cmp>
std::vector<double> optimize::calc_w(graph<cmp>& g,std::vector<sample_data> samples,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
    std::vector<std::vector<array1d<double> > > contracted_vectors; //batched vectors
    size_t n_samples=samples.size();
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            std::vector<array1d<double> > vec(n_samples);
            for(size_t s=0;s<n_samples;s++){
                array1d<double> vec_e(g.vs()[n].rank());
                for(size_t a=0;a<vec_e.nx();a++){
                    if(a!=samples[s].s()[n]){ //if a==samples[s].s()[n], element is log(1)=0. else log(0)=-inf
                        vec_e.at(a)=log(1e-100);
                    }
                }
                vec[s]=vec_e;
            }
            contracted_vectors.push_back(vec);
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(std::vector<array1d<double> >(n_samples));
        }
        l_env.push_back(std::vector<array1d<double> >(n_samples));
        r_env.push_back(std::vector<array1d<double> >(n_samples));
        u_env.push_back(std::vector<array1d<double> >(n_samples));
    }
    size_t contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()][0].nx()!=0)&&(contracted_vectors[(*it).v2()][0].nx()!=0)&&(contracted_vectors[(*it).order()][0].nx()==0)){ //process if children have been contracted and parent is not yet contracted
                std::vector<array1d<double> > res_vec(n_samples,array1d<double>(g.vs()[(*it).order()].rank()));
                for(size_t s=0;s<n_samples;s++){
                    for(size_t k=0;k<(*it).w().nz();k++){
                        std::vector<double> res_vec_addends;
                        for(size_t i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                            for(size_t j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                                res_vec_addends.push_back(contracted_vectors[(*it).v1()][s].at(i)+contracted_vectors[(*it).v2()][s].at(j)+(*it).w().at(i,j,k)); //log space
                            }
                        }
                        res_vec[s].at(k)=lse(res_vec_addends); //log space
                    }
                }
                contracted_vectors[(*it).order()]=res_vec;
                l_env[(*it).order()]=contracted_vectors[(*it).v1()];
                r_env[(*it).order()]=contracted_vectors[(*it).v2()];
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            if((u_env[(*it).v1()][0].nx()==0)&&(u_env[(*it).v2()][0].nx()==0)){
                if(contracted_idx_count==0){ //top tensor's u_env is all ones
                    u_env[(*it).order()]=std::vector<array1d<double> >(n_samples,array1d<double>(g.vs()[(*it).order()].rank())); //0 because log(1)=0
                }
                std::vector<array1d<double> > res_vec_l(n_samples,array1d<double>(g.vs()[(*it).v1()].rank()));
                for(size_t s=0;s<n_samples;s++){
                    for(size_t i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                        std::vector<double> res_vec_l_addends;
                        for(size_t j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                res_vec_l_addends.push_back(r_env[(*it).order()][s].at(j)+u_env[(*it).order()][s].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                        }
                        res_vec_l[s].at(i)=lse(res_vec_l_addends); //log space
                    }
                }
                u_env[(*it).v1()]=res_vec_l;
                std::vector<array1d<double> > res_vec_r(n_samples,array1d<double>(g.vs()[(*it).v2()].rank()));
                for(size_t s=0;s<n_samples;s++){
                    for(size_t j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                        std::vector<double> res_vec_r_addends;
                        for(size_t i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                res_vec_r_addends.push_back(l_env[(*it).order()][s].at(i)+u_env[(*it).order()][s].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                        }
                        res_vec_r[s].at(j)=lse(res_vec_r_addends); //log space
                    }
                }
                u_env[(*it).v2()]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    std::vector<double> w(n_samples);
    for(size_t s=0;s<n_samples;s++){
        std::vector<double> w_addends;
        for(size_t k=0;k<g.vs()[g.vs().size()-1].rank();k++){
            w_addends.push_back(contracted_vectors[g.vs().size()-1][s].at(k));
        }
        w[s]=lse(w_addends);
    }
    return w;
}
template std::vector<double> optimize::calc_w(graph<bmi_comparator>&,std::vector<sample_data>,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);