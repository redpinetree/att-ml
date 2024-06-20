#include "algorithm_nll.hpp"
#include "optimize_nll.hpp"
#include "ttn_ops.hpp"

template<typename cmp>
double optimize::opt_nll(graph<cmp>& g,std::vector<sample_data>& samples,size_t iter_max){
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
        double z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
        std::vector<array3d<double> > dz=calc_dz(l_env_z,r_env_z,u_env_z); //index i corresponds to tensor with order i so some (for input sites) are empty
        
        std::vector<std::vector<array1d<double> > > l_env_sample;
        std::vector<std::vector<array1d<double> > > r_env_sample;
        std::vector<std::vector<array1d<double> > > u_env_sample;
        std::vector<double> w=calc_w(g,samples,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        std::vector<std::vector<array3d<double> > > dw=calc_dw(l_env_sample,r_env_sample,u_env_sample); //index i corresponds to tensor with order i so some (for input sites) are empty
        
        std::multiset<bond,bmi_comparator> new_es;
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond current=*it;
            size_t n=(*it).order();
            size_t shifted_n=n-g.n_phys_sites();
            array3d<double> grad(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_z_term(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_w_term(current.w().nx(),current.w().ny(),current.w().nz());
            // std::cout<<"n="<<n<<"\n";
            // std::cout<<(std::string) current.w()<<"\n";
            std::vector<double> sum_addends;
            for(size_t i=0;i<grad_z_term.nx();i++){
                for(size_t j=0;j<grad_z_term.ny();j++){
                    for(size_t k=0;k<grad_z_term.nz();k++){
                        grad_z_term.at(i,j,k)=dz[n].at(i,j,k)-z; //log space
                        std::vector<double> grad_w_term_addends;
                        for(size_t s=0;s<n_samples;s++){
                            grad_w_term_addends.push_back(dw[n][s].at(i,j,k)-w[s]); //log space
                        }
                        // std::cout<<lse(grad_w_term_addends)<<" "<<log(n_samples)<<"\n";
                        grad_w_term.at(i,j,k)=lse(grad_w_term_addends)-log(n_samples);
                        // grad.at(i,j,k)=(exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k)))*exp(current.w().at(i,j,k));
                        // current.w().at(i,j,k)-=0.1*grad.at(i,j,k);
                        grad.at(i,j,k)=(exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k)))*exp(current.w().at(i,j,k));
                        m[shifted_n].at(i,j,k)=(beta1*m[shifted_n].at(i,j,k))+((1-beta1)*grad.at(i,j,k));
                        v[shifted_n].at(i,j,k)=(beta2*v[shifted_n].at(i,j,k))+((1-beta2)*grad.at(i,j,k)*grad.at(i,j,k));
                        double corrected_m=m[shifted_n].at(i,j,k)/(1-pow(beta1,(double) t));
                        double corrected_v=v[shifted_n].at(i,j,k)/(1-pow(beta2,(double) t));
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
template double optimize::opt_nll(graph<bmi_comparator>&,std::vector<sample_data>&,size_t);

template<typename cmp>
double optimize::hopt_nll(graph<cmp>& g,size_t n_samples,size_t iter_max){
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
    
    size_t current_depth=1;
    size_t max_depth=1;
    auto it=g.es().begin();
    std::vector<sample_data> samples;
    while(it!=g.es().end()){ //due to comparator, bonds are already sorted by depth
        bond current=*it;
        current_depth=(*it).depth();
        size_t n=(*it).order();
        size_t shifted_n=n-g.n_phys_sites();
        
        //construct sample data, only when considering tensors at max layer or at next layer
        if(current_depth>=max_depth){
            std::cout<<"constructing new dataset rooted at tensor "<<n<<"\n";
            samples=sampling::mh_sample(n,g,n_samples); //consider subtree rooted at n
        }
        
        if(current_depth>max_depth){ //if bond is in next layer, reset layer counter and increment maximum layer
            current_depth=1;
            max_depth++;
            it=g.es().begin();
            std::cout<<"next layer "<<max_depth<<"\n";
            continue;
        }
        
        std::vector<array1d<double> > l_env_z;
        std::vector<array1d<double> > r_env_z;
        std::vector<array1d<double> > u_env_z;
        double z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
        std::vector<array3d<double> > dz=calc_dz(l_env_z,r_env_z,u_env_z); //index i corresponds to tensor with order i so some (for input sites) are empty
        
        std::vector<std::vector<array1d<double> > > l_env_sample;
        std::vector<std::vector<array1d<double> > > r_env_sample;
        std::vector<std::vector<array1d<double> > > u_env_sample;
        std::vector<double> w=calc_w(g,samples,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        std::vector<std::vector<array3d<double> > > dw=calc_dw(l_env_sample,r_env_sample,u_env_sample); //index i corresponds to tensor with order i so some (for input sites) are empty
        double prev_nll=1e50;
        for(size_t t=1;t<=iter_max;t++){
            array3d<double> grad(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_z_term(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_w_term(current.w().nx(),current.w().ny(),current.w().nz());
            // std::cout<<"n="<<n<<"\n";
            // std::cout<<(std::string) current.w()<<"\n";
            std::vector<double> sum_addends;
            for(size_t i=0;i<grad_z_term.nx();i++){
                for(size_t j=0;j<grad_z_term.ny();j++){
                    for(size_t k=0;k<grad_z_term.nz();k++){
                        grad_z_term.at(i,j,k)=dz[n].at(i,j,k)-z; //log space
                        std::vector<double> grad_w_term_addends;
                        for(size_t s=0;s<n_samples;s++){
                            grad_w_term_addends.push_back(dw[n][s].at(i,j,k)-w[s]); //log space
                        }
                        // std::cout<<lse(grad_w_term_addends)<<" "<<log(n_samples)<<"\n";
                        grad_w_term.at(i,j,k)=lse(grad_w_term_addends)-log(n_samples);
                        // grad.at(i,j,k)=(exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k)))*exp(current.w().at(i,j,k));
                        // current.w().at(i,j,k)-=0.1*grad.at(i,j,k);
                        grad.at(i,j,k)=(exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k)))*exp(current.w().at(i,j,k));
                        m[shifted_n].at(i,j,k)=(beta1*m[shifted_n].at(i,j,k))+((1-beta1)*grad.at(i,j,k));
                        v[shifted_n].at(i,j,k)=(beta2*v[shifted_n].at(i,j,k))+((1-beta2)*grad.at(i,j,k)*grad.at(i,j,k));
                        double corrected_m=m[shifted_n].at(i,j,k)/(1-pow(beta1,(double) t));
                        double corrected_v=v[shifted_n].at(i,j,k)/(1-pow(beta2,(double) t));
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
            
            //update z
            std::vector<double> z_addends;
            for(size_t i=0;i<l_env_z[n].nx();i++){
                for(size_t j=0;j<r_env_z[n].nx();j++){
                    for(size_t k=0;k<u_env_z[n].nx();k++){
                        z_addends.push_back(current.w().at(i,j,k)+l_env_z[n].at(i)+r_env_z[n].at(j)+u_env_z[n].at(k));
                    }
                }
            }
            z=lse(z_addends);
            //update w[s]
            for(size_t s=0;s<w.size();s++){
                std::vector<double> w_addends;
                for(size_t i=0;i<l_env_sample[n][s].nx();i++){
                    for(size_t j=0;j<r_env_sample[n][s].nx();j++){
                        for(size_t k=0;k<u_env_sample[n][s].nx();k++){
                            w_addends.push_back(current.w().at(i,j,k)+l_env_sample[n][s].at(i)+r_env_sample[n][s].at(j)+u_env_sample[n][s].at(k));
                        }
                    }
                }
                w[s]=lse(w_addends);
            }
            
            //calculate nll and check for convergence
            nll=0;
            for(size_t s=0;s<n_samples;s++){
                nll-=w[s]; //w[s] is log(w(s))
            }
            nll/=(double) n_samples;
            nll+=z; //z is log(z)
            if(fabs(prev_nll-nll)<1e-8){
                std::cout<<"NLL optimization converged after "<<t<<" iterations.\n";
                std::cout<<"nll="<<nll<<" for tensor "<<n<<", with max depth "<<max_depth<<"\n";
                break;
            }
            else{
                if(((t-1)%100)==0){
                    std::cout<<"nll="<<nll<<" for tensor "<<n<<", with max depth "<<max_depth<<"\n";
                }
            }
            prev_nll=nll;
        }
        
        g.es().erase(g.es().lower_bound(current)); //remove element
        g.es().insert(current); //reinsert element
        algorithm::calculate_site_probs(g,current);
        it=g.es().upper_bound(current); //point iterator to next element
    }
    
    return nll;
}
template double optimize::hopt_nll(graph<bmi_comparator>&,size_t,size_t);