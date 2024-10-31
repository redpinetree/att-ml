#include <limits>
#include <list>
#include <random>

#include "omp.h"

#include "algorithm_nll.hpp"
#include "mat_ops.hpp"
#include "mpi_utils.hpp"
#include "optimize_nll.hpp"
#include "ttn_ops.hpp"


//Condat's algorithm (O(n)) https://optimization-online.org/wp-content/uploads/2014/08/4498.pdf
array3d<double> proj_probability_simplex(array3d<double>& w){
    array3d<double> res=w;
    std::vector<double> e=w.e();
    
    std::list<double> l;
    std::list<double> l_aux;
    l.push_back(e[0]);
    double rho=e[0]-1;
    for(size_t n=1;n<e.size();n++){
        if(e[n]>rho){
            rho+=(e[n]-rho)/(double) (l.size()+1);
            if(rho>(e[n]-1)){
                l.push_back(e[n]);
            }
            else{
                l_aux.splice(l_aux.begin(),l);
                l.clear();
                l.push_back(e[n]);
                double rho=e[n]-1;
            }
        }
    }
    if(l_aux.size()!=0){
        for(auto it_aux=l_aux.begin();it_aux!=l_aux.end();++it_aux){
            if((*it_aux)>rho){
                l.push_back(*it_aux);
                rho+=((*it_aux)-rho)/(double) l.size();
            }
        }
    }
    size_t old_l_size;
    do{
        old_l_size=l.size();
        auto it=l.begin();
        while(it!=l.end()){
            if((*it)<=rho){
                rho+=(rho-(*it))/(double) l.size();
                it=l.erase(it); //it is already at next position
            }
            else{
                ++it; //bring it to next position
            }
        }
    }
    while(old_l_size!=l.size());
    double tau=rho;
    
    for(size_t i=0;i<res.nx();i++){
        for(size_t j=0;j<res.ny();j++){
            for(size_t k=0;k<res.nz();k++){
                res.at(i,j,k)=log(((res.at(i,j,k)-tau)>1e-100)?(res.at(i,j,k)-tau):1e-100);
            }
        }
    }
    return res;
}

array4d<double> proj_probability_simplex(array4d<double>& w){
    array4d<double> res=w;
    std::vector<double> e=w.e();
    
    std::list<double> l;
    std::list<double> l_aux;
    l.push_back(e[0]);
    double rho=e[0]-1;
    for(size_t n=1;n<e.size();n++){
        if(e[n]>rho){
            rho+=(e[n]-rho)/(double) (l.size()+1);
            if(rho>(e[n]-1)){
                l.push_back(e[n]);
            }
            else{
                l_aux.splice(l_aux.begin(),l);
                l.clear();
                l.push_back(e[n]);
                double rho=e[n]-1;
            }
        }
    }
    if(l_aux.size()!=0){
        for(auto it_aux=l_aux.begin();it_aux!=l_aux.end();++it_aux){
            if((*it_aux)>rho){
                l.push_back(*it_aux);
                rho+=((*it_aux)-rho)/(double) l.size();
            }
        }
    }
    size_t old_l_size;
    do{
        old_l_size=l.size();
        auto it=l.begin();
        while(it!=l.end()){
            if((*it)<=rho){
                rho+=(rho-(*it))/(double) l.size();
                it=l.erase(it); //it is already at next position
            }
            else{
                ++it; //bring it to next position
            }
        }
    }
    while(old_l_size!=l.size());
    double tau=rho;
    
    for(size_t i=0;i<res.nx();i++){
        for(size_t j=0;j<res.ny();j++){
            for(size_t k=0;k<res.nz();k++){
                for(size_t l=0;l<res.nw();l++){
                    res.at(i,j,k,l)=log(((res.at(i,j,k,l)-tau)>1e-100)?(res.at(i,j,k,l)-tau):1e-100);
                }
            }
        }
    }
    return res;
}

//O(n log n) algorithm
array3d<double> proj_probability_simplex2(array3d<double>& w){
    array3d<double> res=w;
    std::vector<double> e=w.e();
    struct{
        bool operator()(double a,double b){return a>b;}
    } greater_cmp;
    std::sort(e.begin(),e.end(),greater_cmp);
    double rho=0;
    double sum_up_to_rho=0;
    double sum_up_to_a=0;
    for(size_t a=0;a<e.size();a++){
        sum_up_to_a+=e[a];
        double val=e[a]+((1-sum_up_to_a)/(double) (a+1));
        // double val=e[a]+((1-sum_up_to_a)/(double) a);
        if(rho<val){
            rho=val;
            sum_up_to_rho=sum_up_to_a;
        }
    }
    double lambda=(1-sum_up_to_rho)/rho;
    for(size_t i=0;i<res.nx();i++){
        for(size_t j=0;j<res.ny();j++){
            for(size_t k=0;k<res.nz();k++){
                res.at(i,j,k)=log(((res.at(i,j,k)+lambda)>1e-100)?(res.at(i,j,k)+lambda):1e-100);
            }
        }
    }
    return res;
}

void aux_update_lr_cache(bond& current,bond& parent,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample){
    array1d<double> res_vec_z(current.w().nz());
    for(size_t k=0;k<current.w().nz();k++){
        std::vector<double> res_vec_z_addends;
        for(size_t i=0;i<current.w().nx();i++){
            for(size_t j=0;j<current.w().ny();j++){
                res_vec_z_addends.push_back(l_env_z[current.order()].at(i)+r_env_z[current.order()].at(j)+current.w().at(i,j,k)); //log space
            }
        }
        res_vec_z.at(k)=lse(res_vec_z_addends); //log space
    }
    if(parent.v1()==current.order()){
        l_env_z[parent.order()]=res_vec_z;
    }
    else{
        r_env_z[parent.order()]=res_vec_z;
    }
    #pragma omp parallel for
    for(size_t s=0;s<l_env_sample[current.order()].size();s++){
        array1d<double> res_vec_sample(current.w().nz());
        for(size_t k=0;k<current.w().nz();k++){
            std::vector<double> res_vec_sample_addends;
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    res_vec_sample_addends.push_back(l_env_sample[current.order()][s].at(i)+r_env_sample[current.order()][s].at(j)+current.w().at(i,j,k)); //log space
                }
            }
            res_vec_sample.at(k)=lse(res_vec_sample_addends); //log space
        }
        if(parent.v1()==current.order()){
            l_env_sample[parent.order()][s]=res_vec_sample;
        }
        else{
            r_env_sample[parent.order()][s]=res_vec_sample;
        }
    }
}

void aux_update_u_cache(bond& current,bond& parent,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample){
    array1d<double> res_vec2_z(current.w().nz());
    if(parent.v1()==current.order()){
        for(size_t i=0;i<parent.w().nx();i++){
            std::vector<double> res_vec2_z_addends;
            for(size_t j=0;j<parent.w().ny();j++){
                for(size_t k=0;k<parent.w().nz();k++){
                    res_vec2_z_addends.push_back(r_env_z[parent.order()].at(j)+u_env_z[parent.order()].at(k)+parent.w().at(i,j,k)); //log space
                }
            }
            res_vec2_z.at(i)=lse(res_vec2_z_addends); //log space
        }
    }
    else{
        for(size_t j=0;j<parent.w().ny();j++){
            std::vector<double> res_vec2_z_addends;
            for(size_t i=0;i<parent.w().nx();i++){
                for(size_t k=0;k<parent.w().nz();k++){
                    res_vec2_z_addends.push_back(l_env_z[parent.order()].at(i)+u_env_z[parent.order()].at(k)+parent.w().at(i,j,k)); //log space
                }
            }
            res_vec2_z.at(j)=lse(res_vec2_z_addends); //log space
        }
    }
    u_env_z[current.order()]=res_vec2_z;
    #pragma omp parallel for
    for(size_t s=0;s<u_env_sample[parent.order()].size();s++){
        array1d<double> res_vec2_sample(current.w().nz());
        if(parent.v1()==current.order()){
            for(size_t i=0;i<parent.w().nx();i++){
                std::vector<double> res_vec2_sample_addends;
                for(size_t j=0;j<parent.w().ny();j++){
                    for(size_t k=0;k<parent.w().nz();k++){
                        res_vec2_sample_addends.push_back(r_env_sample[parent.order()][s].at(j)+u_env_sample[parent.order()][s].at(k)+parent.w().at(i,j,k)); //log space
                    }
                }
                res_vec2_sample.at(i)=lse(res_vec2_sample_addends); //log space
            }
        }
        else{
            for(size_t j=0;j<parent.w().ny();j++){
                std::vector<double> res_vec2_sample_addends;
                for(size_t i=0;i<parent.w().nx();i++){
                    for(size_t k=0;k<parent.w().nz();k++){
                        res_vec2_sample_addends.push_back(l_env_sample[parent.order()][s].at(i)+u_env_sample[parent.order()][s].at(k)+parent.w().at(i,j,k)); //log space
                    }
                }
                res_vec2_sample.at(j)=lse(res_vec2_sample_addends); //log space
            }
        }
        u_env_sample[current.order()][s]=res_vec2_sample;
    }
}

template<typename cmp>
double optimize::opt_nll(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t iter_max,double lr){
    if(iter_max==0){return 0;}
    double prev_nll=1e50;
    double nll=0;
    //adam variables
    // double lr=0.005;
    double beta1=0.9;
    double beta2=0.999;
    double epsilon=1e-8;
    //initialize adam m,v caches
    std::map<size_t,array3d<double> > m;
    std::map<size_t,array3d<double> > v;
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        size_t n=current.order();
        m[n]=array3d<double>(current.w().nx(),current.w().ny(),current.w().nz());
        v[n]=array3d<double>(current.w().nx(),current.w().ny(),current.w().nz());
    }
    
    std::vector<array1d<double> > l_env_z;
    std::vector<array1d<double> > r_env_z;
    std::vector<array1d<double> > u_env_z;
    std::vector<std::vector<array1d<double> > > l_env_sample;
    std::vector<std::vector<array1d<double> > > r_env_sample;
    std::vector<std::vector<array1d<double> > > u_env_sample;
    double z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
    std::vector<array3d<double> > dz=calc_dz(l_env_z,r_env_z,u_env_z); //index i corresponds to tensor with order i so some (for input sites) are empty
    std::vector<double> w=calc_w(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
    std::vector<std::vector<array3d<double> > > dw=calc_dw(l_env_sample,r_env_sample,u_env_sample); //index i corresponds to tensor with order i so some (for input sites) are empty
    
    double best_nll=1e50;
    std::multiset<bond,bmi_comparator> best_es;
    for(size_t t=1;t<=iter_max;t++){
        
        std::multiset<bond,bmi_comparator> new_es;
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond current=*it;
            size_t n=(*it).order();
            array3d<double> grad(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_z_term(current.w().nx(),current.w().ny(),current.w().nz());
            array3d<double> grad_w_term(current.w().nx(),current.w().ny(),current.w().nz());
            // std::cout<<"n="<<n<<"\n";
            // std::cout<<(std::string) current.w()<<"\n";
            // std::vector<double> sum_addends;
            for(size_t i=0;i<grad_z_term.nx();i++){
                for(size_t j=0;j<grad_z_term.ny();j++){
                    for(size_t k=0;k<grad_z_term.nz();k++){
                        grad_z_term.at(i,j,k)=dz[n].at(i,j,k)-z; //log space
                        std::vector<double> grad_w_term_addends;
                        for(size_t s=0;s<samples.size();s++){
                            grad_w_term_addends.push_back(dw[n][s].at(i,j,k)-w[s]); //log space
                        }
                        // std::cout<<lse(grad_w_term_addends)<<" "<<log(samples.size())<<"\n";
                        grad_w_term.at(i,j,k)=lse(grad_w_term_addends)-log(samples.size());
                        
                        //perform unconstrained gd on log-parametrized problem
                        // grad.at(i,j,k)=(exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k)))*exp(current.w().at(i,j,k));
                        // m[n].at(i,j,k)=(beta1*m[n].at(i,j,k))+((1-beta1)*grad.at(i,j,k));
                        // v[n].at(i,j,k)=(beta2*v[n].at(i,j,k))+((1-beta2)*grad.at(i,j,k)*grad.at(i,j,k));
                        // double corrected_m=m[n].at(i,j,k)/(1-pow(beta1,(double) t));
                        // double corrected_v=v[n].at(i,j,k)/(1-pow(beta2,(double) t));
                        // current.w().at(i,j,k)=current.w().at(i,j,k)-(lr*grad.at(i,j,k));
                        // current.w().at(i,j,k)=current.w().at(i,j,k)-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                        // if(current.w().at(i,j,k)-(lr*grad.at(i,j,k))<log(1e-100)){current.w().at(i,j,k)=log(1e-100);}
                        // else{current.w().at(i,j,k)-=lr*grad.at(i,j,k);}
                        // if(current.w().at(i,j,k)-(lr*grad.at(i,j,k))<log(1e-100)){current.w().at(i,j,k)=log(1e-100);}
                        // else{current.w().at(i,j,k)-=lr*(corrected_m/(sqrt(corrected_v)+epsilon));}
                        
                        //perform unconstrained gd on original problem
                        // grad.at(i,j,k)=exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k));
                        if(grad_z_term.at(i,j,k)>grad_w_term.at(i,j,k)){ //more accurate calculation
                            grad.at(i,j,k)=exp(grad_z_term.at(i,j,k)+log1p(-exp(-(grad_z_term.at(i,j,k)-grad_w_term.at(i,j,k)))));
                        }
                        else{
                            grad.at(i,j,k)=-exp(grad_w_term.at(i,j,k)+log1p(-exp(-(grad_w_term.at(i,j,k)-grad_z_term.at(i,j,k)))));
                        }
                        
                        m[n].at(i,j,k)=(beta1*m[n].at(i,j,k))+((1-beta1)*grad.at(i,j,k));
                        v[n].at(i,j,k)=(beta2*v[n].at(i,j,k))+((1-beta2)*grad.at(i,j,k)*grad.at(i,j,k));
                        double corrected_m=m[n].at(i,j,k)/(1-pow(beta1,(double) t));
                        double corrected_v=v[n].at(i,j,k)/(1-pow(beta2,(double) t));
                        // current.w().at(i,j,k)=exp(current.w().at(i,j,k))-(lr*grad.at(i,j,k));
                        current.w().at(i,j,k)=exp(current.w().at(i,j,k))-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                        // if((exp(current.w().at(i,j,k))-(lr*grad.at(i,j,k)))<0){current.w().at(i,j,k)=1e-100;}
                        // else{current.w().at(i,j,k)=exp(current.w().at(i,j,k))-(lr*grad.at(i,j,k));}
                        // if((exp(current.w().at(i,j,k))-(lr*(corrected_m/(sqrt(corrected_v)+epsilon))))<0){current.w().at(i,j,k)=1e-100;}
                        // else{current.w().at(i,j,k)=exp(current.w().at(i,j,k))-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));}
                        
                        // std::cout<<current.w().at(i,j,k)<<"\n";
                        // sum_addends.push_back(current.w().at(i,j,k));
                    }
                }
            }
            // double sum=lse(sum_addends);
            // for(size_t i=0;i<current.w().nx();i++){
                // for(size_t j=0;j<current.w().ny();j++){
                    // for(size_t k=0;k<current.w().nz();k++){
                        // current.w().at(i,j,k)-=sum;
                        // if(current.w().at(i,j,k)<log(1e-100)){current.w().at(i,j,k)=log(1e-100);}
                    // }
                // }
            // }
            
            //project solution onto probability simplex
            // std::cout<<(std::string) current.w()<<"\n";
            current.w()=proj_probability_simplex(current.w());
            // std::cout<<(std::string) current.w()<<"\n";
            
            // std::cout<<(std::string) grad_z_term.exp_form()<<"\n";
            // std::cout<<(std::string) grad_w_term.exp_form()<<"\n";
            // std::cout<<(std::string) grad<<"\n";
            // double grad_norm=0;
            // for(size_t i=0;i<grad.nx();i++){
                // for(size_t j=0;j<grad.ny();j++){
                    // for(size_t k=0;k<grad.nz();k++){
                        // grad_norm+=grad.at(i,j,k)*grad.at(i,j,k);
                    // }
                // }
            // }
            // grad_norm=sqrt(grad_norm);
            // std::cout<<n<<": "<<grad_norm<<"\n";
            new_es.insert(current);
        }
        
        g.es()=new_es;
        // std::cout<<"new\n";
        // for(auto it=g.es().begin();it!=g.es().end();++it){
            // std::cout<<(std::string) (*it).w()<<"\n";
        // }
        
        z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
        dz=calc_dz(l_env_z,r_env_z,u_env_z); //index i corresponds to tensor with order i so some (for input sites) are empty
        w=calc_w(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        dw=calc_dw(l_env_sample,r_env_sample,u_env_sample); //index i corresponds to tensor with order i so some (for input sites) are empty
        
        //calculate nll and check for convergence
        nll=0;
        for(size_t s=0;s<samples.size();s++){
            nll-=w[s]; //w[s] is log(w(s))
        }
        nll/=(double) samples.size();
        nll+=z; //z is log(z)
        if(nll<best_nll){
            best_nll=nll;
            best_es=g.es();
        }
        if(fabs(prev_nll-nll)<1e-12){
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
    g.es()=best_es;
    return best_nll;
}
template double optimize::opt_nll(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,double);


template<typename cmp>
double optimize::opt_struct_nll(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<size_t>& train_labels,std::vector<sample_data>& test_samples,std::vector<size_t>& test_labels,size_t iter_max,size_t r_max,bool compress_r,double lr,std::map<size_t,double>& train_nll_history,std::map<size_t,double>& test_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
    if(iter_max==0){return 0;}
    size_t single_site_update_count=10;
    double prev_nll=1e50;
    double nll=0;
    double test_nll=0;
    //adam variables
    // double lr=0.001;
    double beta1=0.9;
    double beta2=0.999;
    double epsilon=1e-8;
    //initialize adam m,v caches
    std::map<std::pair<size_t,size_t>,array4d<double> > fused_m;
    std::map<std::pair<size_t,size_t>,array4d<double> > fused_v;
    for(auto it=g.es().begin();it!=--g.es().end();++it){
        bond current=*it;
        bond parent=g.vs()[g.vs()[(*it).order()].u_idx()].p_bond();
        std::pair<size_t,size_t> key=(current.order()<parent.order())?std::make_pair(current.order(),parent.order()):std::make_pair(parent.order(),current.order());
        fused_m[key]=array4d<double>(current.w().nx(),current.w().ny(),(parent.v1()==current.order())?parent.w().ny():parent.w().nx(),parent.w().nz());
        fused_v[key]=array4d<double>(current.w().nx(),current.w().ny(),(parent.v1()==current.order())?parent.w().ny():parent.w().nx(),parent.w().nz());
    }
    
    std::vector<array1d<double> > l_env_z;
    std::vector<array1d<double> > r_env_z;
    std::vector<array1d<double> > u_env_z;
    std::vector<std::vector<array1d<double> > > l_env_sample;
    std::vector<std::vector<array1d<double> > > r_env_sample;
    std::vector<std::vector<array1d<double> > > u_env_sample;
    double z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
    std::vector<double> w=calc_w(g,train_samples,train_labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
    
    double best_nll=1e50;
    double best_test_nll=1e50;
    std::uniform_real_distribution<> unif_dist(1e-16,1.0);
    std::vector<site> best_vs;
    std::multiset<bond,bmi_comparator> best_es;
    // for(size_t t=1;t<=iter_max;t++){
    size_t t=1;
    size_t iter=1;
    while(iter<iter_max){
        //all tensors have an upstream tensor except the top, so we can loop over all tensors except the top to go through all bonds
        // std::cout<<"iter "<<t<<" ";
        std::set<size_t> done_idxs; //set to store processed bond idxs
        // std::cout<<(std::string)g<<"\n";
        //TODO: traversal scheme, not by looping over set, but by moving to adjacent bonds in upward an d downward sweep. always traverse downwards (like DFS) when possible and exit loop when all done_idxs.size()==g.es().size()-1
        auto it=g.es().begin();
        while(done_idxs.size()<g.es().size()){
        // for(auto it=g.es().begin();done_idxs.size()<g.es().size();++it){
        // for(auto it=g.es().begin();it!=--g.es().end();++it){
            bond current=*it;
            if(current.order()==g.vs().size()-1){ //handle top tensor
                g.vs()[current.order()].p_bond()=current;
                done_idxs.insert(current.order());
                if(done_idxs.size()==g.es().size()){break;}
                //next bond
                bond key;
                key.todo()=0;
                if((g.vs()[current.v1()].depth()!=0)&&(done_idxs.find(current.v1())==done_idxs.end())){
                    key.order()=current.v1();
                }
                else if((g.vs()[current.v2()].depth()!=0)&&(done_idxs.find(current.v2())==done_idxs.end())){
                    key.order()=current.v2();
                }
                key.depth()=g.vs()[key.order()].depth();
                key.bmi()=-1e50;
                key.virt_count()=2;
                it=g.es().lower_bound(key);
                continue;
            }
            bond key;
            key.todo()=0;
            key.order()=g.vs()[(*it).order()].u_idx();
            key.depth()=g.vs()[key.order()].depth();
            key.bmi()=-1e50;
            key.virt_count()=2;
            auto it_parent=g.es().lower_bound(key);
            // auto it_parent=g.es().find(g.vs()[g.vs()[(*it).order()].u_idx()].p_bond());
            bond parent=*it_parent; //must be a dereferenced pointer to the actual object, not a copy!
            // std::cout<<(std::string) current<<"\n";
            // std::cout<<(std::string) parent<<"\n";
            
            array4d<double> fused=optimize::fused_update(current,parent,z,w,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample,fused_m,fused_v,t,lr,beta1,beta2,epsilon);
            
            //save state of graph, iterators, and caches before proposing changes
            graph<cmp> orig_g=g;
            bond orig_current=current;
            bond orig_parent=parent;
            double orig_z=z;
            std::vector<array1d<double> > orig_l_env_z=l_env_z;
            std::vector<array1d<double> > orig_r_env_z=r_env_z;
            std::vector<array1d<double> > orig_u_env_z=u_env_z;
            std::vector<double> orig_w=w;
            std::vector<std::vector<array1d<double> > > orig_l_env_sample=l_env_sample;
            std::vector<std::vector<array1d<double> > > orig_r_env_sample=r_env_sample;
            std::vector<std::vector<array1d<double> > > orig_u_env_sample=u_env_sample;
            
            double bmi1=way1(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,done_idxs,fused,r_max,compress_r,train_samples,train_labels,single_site_update_count,lr,beta1,beta2,epsilon);
            
            if(struct_opt){
                graph<cmp> way1_g=g;
                bond way1_current=current;
                bond way1_parent=parent;
                double way1_z=z;
                std::vector<array1d<double> > way1_l_env_z=l_env_z;
                std::vector<array1d<double> > way1_r_env_z=r_env_z;
                std::vector<array1d<double> > way1_u_env_z=u_env_z;
                std::vector<double> way1_w=w;
                std::vector<std::vector<array1d<double> > > way1_l_env_sample=l_env_sample;
                std::vector<std::vector<array1d<double> > > way1_r_env_sample=r_env_sample;
                std::vector<std::vector<array1d<double> > > way1_u_env_sample=u_env_sample;
                
                g=orig_g;
                current=orig_current;
                parent=orig_parent;
                z=orig_z;
                l_env_z=orig_l_env_z;
                r_env_z=orig_r_env_z;
                u_env_z=orig_u_env_z;
                w=orig_w;
                l_env_sample=orig_l_env_sample;
                r_env_sample=orig_r_env_sample;
                u_env_sample=orig_u_env_sample;
                it=g.es().find(current);
                it_parent=g.es().find(parent);
                
                double bmi2=way2(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,done_idxs,fused,r_max,compress_r,train_samples,train_labels,single_site_update_count,lr,beta1,beta2,epsilon);
                
                graph<bmi_comparator> way2_g=g;
                bond way2_current=current;
                bond way2_parent=parent;
                double way2_z=z;
                std::vector<array1d<double> > way2_l_env_z=l_env_z;
                std::vector<array1d<double> > way2_r_env_z=r_env_z;
                std::vector<array1d<double> > way2_u_env_z=u_env_z;
                std::vector<double> way2_w=w;
                std::vector<std::vector<array1d<double> > > way2_l_env_sample=l_env_sample;
                std::vector<std::vector<array1d<double> > > way2_r_env_sample=r_env_sample;
                std::vector<std::vector<array1d<double> > > way2_u_env_sample=u_env_sample;
                
                g=orig_g;
                current=orig_current;
                parent=orig_parent;
                z=orig_z;
                l_env_z=orig_l_env_z;
                r_env_z=orig_r_env_z;
                u_env_z=orig_u_env_z;
                w=orig_w;
                l_env_sample=orig_l_env_sample;
                r_env_sample=orig_r_env_sample;
                u_env_sample=orig_u_env_sample;
                it=g.es().find(current);
                it_parent=g.es().find(parent);
                
                double bmi3=way3(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,done_idxs,fused,r_max,compress_r,train_samples,train_labels,single_site_update_count,lr,beta1,beta2,epsilon);
                
                graph<bmi_comparator> way3_g=g;
                bond way3_current=current;
                bond way3_parent=parent;
                double way3_z=z;
                std::vector<array1d<double> > way3_l_env_z=l_env_z;
                std::vector<array1d<double> > way3_r_env_z=r_env_z;
                std::vector<array1d<double> > way3_u_env_z=u_env_z;
                std::vector<double> way3_w=w;
                std::vector<std::vector<array1d<double> > > way3_l_env_sample=l_env_sample;
                std::vector<std::vector<array1d<double> > > way3_r_env_sample=r_env_sample;
                std::vector<std::vector<array1d<double> > > way3_u_env_sample=u_env_sample;
                
                // std::cout<<"bmis: "<<bmi1<<" "<<bmi2<<" "<<bmi3<<"\n";
                
                if((bmi1<=bmi2)&&(bmi1<=bmi3)){
                    g=way1_g;
                    current=way1_current;
                    parent=way1_parent;
                    z=way1_z;
                    l_env_z=way1_l_env_z;
                    r_env_z=way1_r_env_z;
                    u_env_z=way1_u_env_z;
                    w=way1_w;
                    l_env_sample=way1_l_env_sample;
                    r_env_sample=way1_r_env_sample;
                    u_env_sample=way1_u_env_sample;
                    it=g.es().find(current);
                    // std::cout<<"selected bmi1\n";
                }
                else if((bmi2<bmi1)&&(bmi2<=bmi3)){
                    g=way2_g;
                    current=way2_current;
                    parent=way2_parent;
                    z=way2_z;
                    l_env_z=way2_l_env_z;
                    r_env_z=way2_r_env_z;
                    u_env_z=way2_u_env_z;
                    w=way2_w;
                    l_env_sample=way2_l_env_sample;
                    r_env_sample=way2_r_env_sample;
                    u_env_sample=way2_u_env_sample;
                    it=g.es().find(current);
                    // std::cout<<"selected bmi2\n";
                }
                else if((bmi3<bmi1)&&(bmi3<bmi2)){
                    g=way3_g;
                    current=way3_current;
                    parent=way3_parent;
                    z=way3_z;
                    l_env_z=way3_l_env_z;
                    r_env_z=way3_r_env_z;
                    u_env_z=way3_u_env_z;
                    w=way3_w;
                    l_env_sample=way3_l_env_sample;
                    r_env_sample=way3_r_env_sample;
                    u_env_sample=way3_u_env_sample;
                    it=g.es().find(current);
                    // std::cout<<"selected bmi3\n";
                }
            }
            
            bond b=*it;
            g.es().erase(it);
            g.es().insert(b);
            done_idxs.insert(current.order());
            // std::cout<<(std::string)g<<"\n";
            
            //next bond
            key.todo()=0;
            if((g.vs()[current.v1()].depth()!=0)&&(done_idxs.find(current.v1())==done_idxs.end())){
                // std::cout<<"L\n";
                key.order()=current.v1();
            }
            else if((g.vs()[current.v2()].depth()!=0)&&(done_idxs.find(current.v2())==done_idxs.end())){
                // std::cout<<"R\n";
                key.order()=current.v2();
            }
            else{
                // std::cout<<"U\n";
                key.order()=g.vs()[current.order()].u_idx();
            }
            key.depth()=g.vs()[key.order()].depth();
            key.bmi()=-1e50;
            key.virt_count()=2;
            it=g.es().lower_bound(key);
            
            // nll=0;
            // for(size_t s=0;s<train_samples.size();s++){
                // nll-=w[s]; //w[s] is log(w(s))
            // }
            // nll/=(double) train_samples.size();
            // nll+=z; //z is log(z)
            // std::cout<<"inner loop nll="<<nll<<"\n";
            
            //calculate train nll
            nll=0;
            #pragma omp parallel for reduction(-:nll)
            for(size_t s=0;s<train_samples.size();s++){
                nll-=w[s]; //w[s] is log(w(s))
            }
            nll/=(double) train_samples.size();
            nll+=z; //z is log(z)
            train_nll_history.insert(std::pair<size_t,double>(iter,nll));
            
            if(iter==iter_max){break;}
            iter++;
        }
        // std::cout<<(std::string) g<<"\n";
        // std::cout<<"new\n";
        // for(auto it=g.es().begin();it!=g.es().end();++it){
            // std::cout<<(std::string) (*it).w().exp_form()<<"\n";
        // }
            
        //calculate test nll per sweep
        if(test_samples.size()!=0){
            std::vector<std::vector<array1d<double> > > test_l_env_sample;
            std::vector<std::vector<array1d<double> > > test_r_env_sample;
            std::vector<std::vector<array1d<double> > > test_u_env_sample;
            std::vector<double> test_w=calc_w(g,test_samples,test_labels,test_l_env_sample,test_r_env_sample,test_u_env_sample);
            test_nll=0;
            #pragma omp parallel for reduction(-:test_nll)
            for(size_t s=0;s<test_samples.size();s++){
                test_nll-=test_w[s]; //w[s] is log(w(s))
            }
            test_nll/=(double) test_samples.size();
            test_nll+=z; //z is log(z)
            test_nll_history.insert(std::pair<size_t,double>(iter,test_nll));
        }
        
        //check for convergence
        if((test_samples.size()!=0)?(test_nll<best_test_nll):(nll<best_nll)){
            best_nll=nll;
            best_test_nll=test_nll;
            best_vs=g.vs();
            best_es=g.es();
        }
        if(iter==iter_max){
            std::cout<<"Maximum iterations reached ("<<iter<<").\n";
            std::cout<<"iter "<<iter<<" nll="<<nll<<"\n";
            train_nll_history.insert(std::pair<size_t,double>(iter,nll));
            break;
        }
        if(fabs((prev_nll-nll)/nll)<1e-6){
            std::cout<<"NLL optimization converged after "<<iter<<" iterations.\n";
            std::cout<<"sweep "<<t<<" iter "<<iter<<" nll="<<nll<<"\n";
            train_nll_history.insert(std::pair<size_t,double>(iter,nll));
            break;
        }
        else{
            if((done_idxs.size()==g.es().size())&&((t-1)%1)==0){
                if(test_samples.size()!=0){
                    std::cout<<"sweep "<<t<<" iter "<<iter<<" train nll="<<nll<<" test nll="<<test_nll<<"\n";
                }
                else{
                    std::cout<<"sweep "<<t<<" iter "<<iter<<" train nll="<<nll<<"\n";
                }
            }
        }
        prev_nll=nll;
        sweep_history.insert(std::pair<size_t,double>(t,iter));
        t++;
    }
    //calculate final train nll
    // nll=0;
    // #pragma omp parallel for reduction(-:nll)
    // for(size_t s=0;s<train_samples.size();s++){
        // nll-=w[s]; //w[s] is log(w(s))
    // }
    // nll/=(double) train_samples.size();
    // nll+=z; //z is log(z)
    // std::cout<<"final nll="<<nll<<"\n";
    // train_nll_history.insert(std::pair<size_t,double>(iter,nll));
    // if(nll<best_nll){
        // best_nll=nll;
        // best_test_nll=test_nll;
        // best_vs=g.vs();
        // best_es=g.es();
    // }
    std::cout<<"best nll="<<best_nll<<"\n";
    std::cout<<"best test nll="<<best_test_nll<<"\n";
    t++;
    nll=best_nll;
    test_nll=best_test_nll;
    g.vs()=best_vs;
    g.es()=best_es;
    
    z=calc_z(g,l_env_z,r_env_z,u_env_z);
    w=calc_w(g,train_samples,train_labels,l_env_sample,r_env_sample,u_env_sample);
    
    // std::cout<<(std::string)g<<"\n";
    for(size_t i=0;i<g.n_phys_sites();i++){
        // double input_bmi=fabs(optimize::calc_bmi_input(i,g.vs()[i].rank(),train_samples,g.vs()[g.vs()[i].u_idx()].p_bond(),l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample)) //take abs
        double input_bmi=optimize::calc_bmi_input(i,g.vs()[i].rank(),train_samples,g.vs()[g.vs()[i].u_idx()].p_bond(),l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
        // std::cout<<input_bmi<<"\n";
        if(input_bmi<-log(g.vs()[i].rank())){input_bmi=2*log(g.vs()[i].rank());}
        g.vs()[i].bmi()=input_bmi;
        // g.vs()[i].bmi()=0;
    }
    
    return best_nll;
}
template double optimize::opt_struct_nll(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
double optimize::hopt_nll(graph<cmp>& g,size_t n_samples,size_t n_sweeps,size_t iter_max){
    double nll=0;
    //adam variables
    double alpha=0.01;
    double beta1=0.9;
    double beta2=0.999;
    double epsilon=1e-8;
    //initialize adam m,v caches
    std::map<size_t,array3d<double> > m;
    std::map<size_t,array3d<double> > v;
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        size_t n=current.order();
        m[n]=array3d<double>(current.w().nx(),current.w().ny(),current.w().nz());
        v[n]=array3d<double>(current.w().nx(),current.w().ny(),current.w().nz());
    }
    
    size_t current_depth=1;
    size_t max_depth=1;
    auto it=g.es().begin();
    std::vector<sample_data> sym_samples;
    std::multiset<bond,bmi_comparator> new_es;
    while(it!=g.es().end()){ //due to comparator, bonds are already sorted by depth
        bond current=*it;
        current_depth=(*it).depth();
        size_t n=(*it).order();
        
        //construct sample data, only when considering tensors at max layer or at next layer
        if(current_depth>=max_depth){
            std::cout<<"constructing new dataset rooted at tensor "<<n<<"\n";
            // std::vector<sample_data> samples=sampling::local_mh_sample(g,n_samples,n_sweeps,0); //consider subtree rooted at n
            std::vector<sample_data> samples=sampling::hybrid_mh_sample(g,n_samples,n_sweeps,0); //consider subtree rooted at n
            //symmetrize samples
            sym_samples=sampling::symmetrize_samples(samples);
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
        std::vector<double> w=calc_w(g,sym_samples,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
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
                        for(size_t s=0;s<sym_samples.size();s++){
                            grad_w_term_addends.push_back(dw[n][s].at(i,j,k)-w[s]); //log space
                        }
                        // std::cout<<lse(grad_w_term_addends)<<" "<<log(sym_samples.size())<<"\n";
                        grad_w_term.at(i,j,k)=lse(grad_w_term_addends)-log(sym_samples.size());
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
            for(size_t s=0;s<sym_samples.size();s++){
                nll-=w[s]; //w[s] is log(w(s))
            }
            nll/=(double) sym_samples.size();
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
template double optimize::hopt_nll(graph<bmi_comparator>&,size_t,size_t,size_t);

template<typename cmp>
double optimize::hopt_nll2(graph<cmp>& g,size_t n_samples,size_t n_sweeps,size_t iter_max){
    double nll=0;
    //adam variables
    double alpha=0.01;
    double beta1=0.9;
    double beta2=0.999;
    double epsilon=1e-8;
    //initialize adam m,v caches
    std::map<size_t,array3d<double> > m;
    std::map<size_t,array3d<double> > v;
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        size_t n=current.order();
        m[n]=array3d<double>(current.w().nx(),current.w().ny(),current.w().nz());
        v[n]=array3d<double>(current.w().nx(),current.w().ny(),current.w().nz());
    }
    
    size_t current_depth=1;
    size_t max_depth=1;
    auto it=g.es().begin();
    // std::vector<sample_data> samples=sampling::local_mh_sample(g,n_samples,n_sweeps,0); //consider subtree rooted at n
    std::vector<sample_data> samples=sampling::hybrid_mh_sample(g,n_samples,n_sweeps,0); //consider subtree rooted at n
    std::vector<sample_data> sym_samples=sampling::symmetrize_samples(samples);
    std::multiset<bond,bmi_comparator> new_es;
    while(it!=g.es().end()){ //due to comparator, bonds are already sorted by depth
        bond current=*it;
        current_depth=(*it).depth();
        size_t n=(*it).order();
        size_t shifted_n=n-g.n_phys_sites();
        
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
        std::vector<double> w=calc_w(g,sym_samples,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
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
                        for(size_t s=0;s<sym_samples.size();s++){
                            grad_w_term_addends.push_back(dw[n][s].at(i,j,k)-w[s]); //log space
                        }
                        // std::cout<<lse(grad_w_term_addends)<<" "<<log(sym_samples.size())<<"\n";
                        grad_w_term.at(i,j,k)=lse(grad_w_term_addends)-log(sym_samples.size());
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
            for(size_t s=0;s<sym_samples.size();s++){
                nll-=w[s]; //w[s] is log(w(s))
            }
            nll/=(double) sym_samples.size();
            nll+=z; //z is log(z)
            if(fabs(prev_nll-nll)<1e-8){
                std::cout<<"NLL optimization converged after "<<t<<" iterations.\n";
                std::cout<<"nll="<<nll<<" for tensor "<<n<<", with max depth "<<max_depth<<"\n";
                break;
            }
            else{
                if(((t-1)%100)==0){
                    // std::cout<<"nll="<<nll<<" for tensor "<<n<<", with max depth "<<max_depth<<"\n";
                }
            }
            prev_nll=nll;
        }
        
        g.es().erase(g.es().lower_bound(current)); //remove element
        g.es().insert(current); //reinsert element
        algorithm::calculate_site_probs(g,current);
        
        //update samples according to updated tree
        sampling::update_samples(n,g,samples);
        sym_samples=sampling::symmetrize_samples(samples);
        
        // for(size_t s=0;s<samples.size();s++){
            // for(size_t m=0;m<samples[s].s().size();m++){
                // if(m==g.n_phys_sites()){std::cout<<": ";}
                // std::cout<<samples[s].s()[m]<<" ";
            // }
            // std::cout<<"\n";
        // }
        // std::cout<<"\n";
        
        it=g.es().upper_bound(current); //point iterator to next element
    }
    
    return nll;
}
template double optimize::hopt_nll2(graph<bmi_comparator>&,size_t,size_t,size_t);

template<typename cmp>
std::vector<size_t> optimize::classify(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<array1d<double> >& probs){
    std::vector<std::vector<array1d<double> > > contracted_vectors; //batched vectors
    size_t n_samples=samples.size();
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            std::vector<array1d<double> > vec(n_samples);
            #pragma omp parallel for
            for(size_t s=0;s<n_samples;s++){
                array1d<double> vec_e(g.vs()[n].rank());
                if(samples[s].s()[n]!=0){
                    for(size_t a=0;a<vec_e.nx();a++){
                        if(a!=(samples[s].s()[n]-1)){ //if a==samples[s].s()[n]-1, element is log(1)=0. else log(0)=-inf
                            vec_e.at(a)=log(1e-100);
                        }
                    }
                }
                vec[s]=vec_e;
            }
            contracted_vectors.push_back(vec);
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(std::vector<array1d<double> >(n_samples));
        }
    }
    size_t contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()][0].nx()!=0)&&(contracted_vectors[(*it).v2()][0].nx()!=0)&&(((*it).order()==g.vs().size()-1)||(contracted_vectors[(*it).order()][0].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                std::vector<array1d<double> > res_vec(n_samples,array1d<double>(g.vs()[(*it).order()].rank()));
                #pragma omp parallel for
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
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    
    bond top_bond=*(g.es().rbegin());
    probs=contracted_vectors[top_bond.order()];
    #pragma omp parallel for
    for(size_t s=0;s<n_samples;s++){
        double prob_sum=lse(probs[s].e());
        for(size_t i=0;i<probs[s].nx();i++){
            probs[s].at(i)-=prob_sum;
        }
    }
    
    std::vector<size_t> classes(n_samples);
    #pragma omp parallel for
    for(size_t s=0;s<n_samples;s++){
        auto max_it=std::max_element(probs[s].e().begin(),probs[s].e().end());
        classes[s]=std::distance(probs[s].e().begin(),max_it);
    }
    return classes;
}
template std::vector<size_t> optimize::classify(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<array1d<double> >&);

void optimize::site_update(bond& b,double z,std::vector<double>& w,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,array3d<double>& m_cache,array3d<double>& v_cache,size_t t,double lr,double beta1,double beta2,double epsilon){
    array3d<double> dz(b.w().nx(),b.w().ny(),b.w().nz());
    std::vector<array3d<double> > dw(w.size());
    for(size_t i=0;i<dz.nx();i++){
        for(size_t j=0;j<dz.ny();j++){
            for(size_t k=0;k<dz.nz();k++){
                dz.at(i,j,k)=l_env_z[b.order()].at(i)+r_env_z[b.order()].at(j)+u_env_z[b.order()].at(k); //log space
            }
        }
    }
    #pragma omp parallel for
    for(size_t s=0;s<w.size();s++){
        array3d<double> dw_e(b.w().nx(),b.w().ny(),b.w().nz());
        for(size_t i=0;i<dw_e.nx();i++){
            for(size_t j=0;j<dw_e.ny();j++){
                for(size_t k=0;k<dw_e.nz();k++){
                    dw_e.at(i,j,k)=l_env_sample[b.order()][s].at(i)+r_env_sample[b.order()][s].at(j)+u_env_sample[b.order()][s].at(k); //log space
                }
            }
        }
        dw[s]=dw_e;
    }

    array3d<double> grad(b.w().nx(),b.w().ny(),b.w().nz());
    array3d<double> grad_z_term(b.w().nx(),b.w().ny(),b.w().nz());
    array3d<double> grad_w_term(b.w().nx(),b.w().ny(),b.w().nz());
    for(size_t i=0;i<grad_z_term.nx();i++){
        for(size_t j=0;j<grad_z_term.ny();j++){
            for(size_t k=0;k<grad_z_term.nz();k++){
                grad_z_term.at(i,j,k)=dz.at(i,j,k)-z; //log space
                std::vector<double> grad_w_term_addends(w.size());
                #pragma omp parallel for
                for(size_t s=0;s<w.size();s++){
                    grad_w_term_addends[s]=dw[s].at(i,j,k)-w[s]; //log space
                }
                grad_w_term.at(i,j,k)=lse(grad_w_term_addends)-log(w.size());
                
                //perform unconstrained gd on original problem
                // grad.at(i,j,k)=exp(grad_z_term.at(i,j,k))-exp(grad_w_term.at(i,j,k));
                if(grad_z_term.at(i,j,k)>grad_w_term.at(i,j,k)){ //more accurate calculation
                    grad.at(i,j,k)=exp(grad_z_term.at(i,j,k)+log1p(-exp(-(grad_z_term.at(i,j,k)-grad_w_term.at(i,j,k)))));
                }
                else{
                    grad.at(i,j,k)=-exp(grad_w_term.at(i,j,k)+log1p(-exp(-(grad_w_term.at(i,j,k)-grad_z_term.at(i,j,k)))));
                }
                
                m_cache.at(i,j,k)=(beta1*m_cache.at(i,j,k))+((1-beta1)*grad.at(i,j,k));
                v_cache.at(i,j,k)=(beta2*v_cache.at(i,j,k))+((1-beta2)*grad.at(i,j,k)*grad.at(i,j,k));
                double corrected_m=m_cache.at(i,j,k)/(1-pow(beta1,(double) t));
                double corrected_v=v_cache.at(i,j,k)/(1-pow(beta2,(double) t));
                b.w().at(i,j,k)=exp(b.w().at(i,j,k))-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                // b.w().at(i,j,k)=exp(b.w().at(i,j,k))-(lr*grad.at(i,j,k));
                if(b.w().at(i,j,k)<0){b.w().at(i,j,k)=1e-300;}
            }
        }
    }
    //project solution onto probability simplex
    // std::cout<<(std::string) b.w()<<"\n";
    // b.w()=proj_probability_simplex(b.w());
    for(size_t i=0;i<b.w().nx();i++){
        for(size_t j=0;j<b.w().ny();j++){
            for(size_t k=0;k<b.w().nz();k++){
                b.w().at(i,j,k)=log(b.w().at(i,j,k));
            }
        }
    }
}

array4d<double> optimize::fused_update(bond& b1,bond& b2,double z,std::vector<double>& w,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,std::map<std::pair<size_t,size_t>,array4d<double> >& m_cache,std::map<std::pair<size_t,size_t>,array4d<double> >& v_cache,size_t t,double lr,double beta1,double beta2,double epsilon){
    //calculate fused, dz and dw
    array4d<double> fused(b1.w().nx(),b1.w().ny(),(b2.v1()==b1.order())?b2.w().ny():b2.w().nx(),b2.w().nz());
    for(size_t i=0;i<fused.nx();i++){
        for(size_t j=0;j<fused.ny();j++){
            for(size_t k=0;k<fused.nz();k++){
                for(size_t l=0;l<fused.nw();l++){
                    std::vector<double> sum_addends;
                    for(size_t m=0;m<b1.w().nz();m++){
                        sum_addends.push_back(b1.w().at(i,j,m)+((b2.v1()==b1.order())?b2.w().at(m,k,l):b2.w().at(k,m,l))); //log space
                    }
                    fused.at(i,j,k,l)=lse(sum_addends);
                }
            }
        }
    }
    
    array4d<double> dz(fused.nx(),fused.ny(),fused.nz(),fused.nw());
    std::vector<array4d<double> > dw(w.size());
    for(size_t i=0;i<dz.nx();i++){
        for(size_t j=0;j<dz.ny();j++){
            for(size_t k=0;k<dz.nz();k++){
                for(size_t l=0;l<dz.nw();l++){
                    dz.at(i,j,k,l)=l_env_z[b1.order()].at(i)+r_env_z[b1.order()].at(j)+((b2.v1()==b1.order())?r_env_z[b2.order()].at(k):l_env_z[b2.order()].at(k))+u_env_z[b2.order()].at(l); //log space
                }
            }
        }
    }
    #pragma omp parallel for
    for(size_t s=0;s<w.size();s++){
        array4d<double> dw_e(fused.nx(),fused.ny(),fused.nz(),fused.nw());
        for(size_t i=0;i<dw_e.nx();i++){
            for(size_t j=0;j<dw_e.ny();j++){
                for(size_t k=0;k<dw_e.nz();k++){
                    for(size_t l=0;l<dw_e.nw();l++){
                        dw_e.at(i,j,k,l)=l_env_sample[b1.order()][s].at(i)+r_env_sample[b1.order()][s].at(j)+((b2.v1()==b1.order())?r_env_sample[b2.order()][s].at(k):l_env_sample[b2.order()][s].at(k))+u_env_sample[b2.order()][s].at(l); //log space
                    }
                }
            }
        }
        dw[s]=dw_e;
    }
    
    std::pair<size_t,size_t> key=(b1.order()<b2.order())?std::make_pair(b1.order(),b2.order()):std::make_pair(b2.order(),b1.order());
    
    //reset m and v caches if there is a change in size
    if((m_cache[key].nx()!=fused.nx())||(m_cache[key].ny()!=fused.ny())||(m_cache[key].nz()!=fused.nz())||(m_cache[key].nw()!=fused.nw())){
        m_cache[key]=array4d<double>(b1.w().nx(),b1.w().ny(),(b2.v1()==b1.order())?b2.w().ny():b2.w().nx(),b2.w().nz());
        v_cache[key]=array4d<double>(b1.w().nx(),b1.w().ny(),(b2.v1()==b1.order())?b2.w().ny():b2.w().nx(),b2.w().nz());
    }
    
    array4d<double> grad_2site(fused.nx(),fused.ny(),fused.nz(),fused.nw());
    array4d<double> grad_z_term_2site(grad_2site.nx(),grad_2site.ny(),grad_2site.nz(),grad_2site.nw());
    array4d<double> grad_w_term_2site(grad_2site.nx(),grad_2site.ny(),grad_2site.nz(),grad_2site.nw());
    // std::cout<<"n="<<n<<"\n";
    // std::cout<<(std::string) b1.w()<<"\n";
    for(size_t i=0;i<grad_2site.nx();i++){
        for(size_t j=0;j<grad_2site.ny();j++){
            for(size_t k=0;k<grad_2site.nz();k++){
                for(size_t l=0;l<grad_2site.nw();l++){
                    grad_z_term_2site.at(i,j,k,l)=dz.at(i,j,k,l)-z; //log space
                    std::vector<double> grad_w_term_2site_addends(w.size());
                    #pragma omp parallel for
                    for(size_t s=0;s<w.size();s++){
                        grad_w_term_2site_addends[s]=dw[s].at(i,j,k,l)-w[s]; //log space
                    }
                    grad_w_term_2site.at(i,j,k,l)=lse(grad_w_term_2site_addends)-log(w.size());
                    
                    //perform projected nonnegative gd on original problem
                    // grad_2site.at(i,j,k)=exp(grad_z_term_2site.at(i,j,k))-exp(grad_w_term_2site.at(i,j,k));
                    if(grad_z_term_2site.at(i,j,k,l)>grad_w_term_2site.at(i,j,k,l)){ //more accurate calculation
                        grad_2site.at(i,j,k,l)=exp(grad_z_term_2site.at(i,j,k,l)+log1p(-exp(-(grad_z_term_2site.at(i,j,k,l)-grad_w_term_2site.at(i,j,k,l)))));
                    }
                    else{
                        grad_2site.at(i,j,k,l)=-exp(grad_w_term_2site.at(i,j,k,l)+log1p(-exp(-(grad_w_term_2site.at(i,j,k,l)-grad_z_term_2site.at(i,j,k,l)))));
                    }
                    m_cache[key].at(i,j,k,l)=(beta1*m_cache[key].at(i,j,k,l))+((1-beta1)*grad_2site.at(i,j,k,l));
                    v_cache[key].at(i,j,k,l)=(beta2*v_cache[key].at(i,j,k,l))+((1-beta2)*grad_2site.at(i,j,k,l)*grad_2site.at(i,j,k,l));
                    double corrected_m=m_cache[key].at(i,j,k,l)/(1-pow(beta1,(double) t));
                    double corrected_v=v_cache[key].at(i,j,k,l)/(1-pow(beta2,(double) t));
                    fused.at(i,j,k,l)=exp(fused.at(i,j,k,l))-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                    // fused.at(i,j,k,l)=exp(fused.at(i,j,k,l))-(lr*grad_2site.at(i,j,k,l));
                    if(fused.at(i,j,k,l)<0){fused.at(i,j,k,l)=1e-300;}
                }
            }
        }
    }
    return fused;
}

double optimize::calc_bmi(bond& current,bond& parent,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample){
    if(current.w().nz()==1){return 0;} //bmi is nonnegative and upper-bounded by bond dimension, so dim=1 -> bmi=0
    size_t n_samples=u_env_sample[0].size();
    array1d<double> a_subsystem_vec_z(current.w().nz());
    array1d<double> b_subsystem_vec_z(current.w().nz());
    std::vector<array1d<double> > a_subsystem_vec_sample(n_samples);
    std::vector<array1d<double> > b_subsystem_vec_sample(n_samples);
    for(size_t k=0;k<current.w().nz();k++){
        std::vector<double> a_subsystem_vec_z_addends;
        for(size_t i=0;i<current.w().nx();i++){
            for(size_t j=0;j<current.w().ny();j++){
                a_subsystem_vec_z_addends.push_back(l_env_z[current.order()].at(i)+r_env_z[current.order()].at(j)+current.w().at(i,j,k));
            }
        }
        a_subsystem_vec_z.at(k)=lse(a_subsystem_vec_z_addends);
    }
    #pragma omp parallel for
    for(size_t s=0;s<n_samples;s++){
        a_subsystem_vec_sample[s]=array1d<double>(current.w().nz());
        for(size_t k=0;k<current.w().nz();k++){
            std::vector<double> a_subsystem_vec_sample_addends;
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    a_subsystem_vec_sample_addends.push_back(l_env_sample[current.order()][s].at(i)+r_env_sample[current.order()][s].at(j)+current.w().at(i,j,k));
                }
            }
            a_subsystem_vec_sample[s].at(k)=lse(a_subsystem_vec_sample_addends);
        }
    }
    if((current.order()==parent.v1())){
        for(size_t k=0;k<parent.w().nx();k++){
            std::vector<double> b_subsystem_vec_z_addends;
            for(size_t l=0;l<parent.w().ny();l++){
                for(size_t m=0;m<parent.w().nz();m++){
                    b_subsystem_vec_z_addends.push_back(r_env_z[parent.order()].at(l)+u_env_z[parent.order()].at(m)+parent.w().at(k,l,m));
                }
            }
            b_subsystem_vec_z.at(k)=lse(b_subsystem_vec_z_addends);
        }
        #pragma omp parallel for
        for(size_t s=0;s<n_samples;s++){
            b_subsystem_vec_sample[s]=array1d<double>(current.w().nz());
            for(size_t k=0;k<parent.w().nx();k++){
                std::vector<double> b_subsystem_vec_sample_addends;
                for(size_t l=0;l<parent.w().ny();l++){
                    for(size_t m=0;m<parent.w().nz();m++){
                        b_subsystem_vec_sample_addends.push_back(r_env_sample[parent.order()][s].at(l)+u_env_sample[parent.order()][s].at(m)+parent.w().at(k,l,m));
                    }
                }
                b_subsystem_vec_sample[s].at(k)=lse(b_subsystem_vec_sample_addends);
            }
        }
    }
    else{
        for(size_t l=0;l<parent.w().ny();l++){
            std::vector<double> b_subsystem_vec_z_addends;
            for(size_t k=0;k<parent.w().nx();k++){
                for(size_t m=0;m<parent.w().nz();m++){
                    b_subsystem_vec_z_addends.push_back(l_env_z[parent.order()].at(k)+u_env_z[parent.order()].at(m)+parent.w().at(k,l,m));
                }
            }
            b_subsystem_vec_z.at(l)=lse(b_subsystem_vec_z_addends);
        }
        #pragma omp parallel for
        for(size_t s=0;s<n_samples;s++){
            b_subsystem_vec_sample[s]=array1d<double>(current.w().nz());
            for(size_t l=0;l<parent.w().ny();l++){
                std::vector<double> b_subsystem_vec_sample_addends;
                for(size_t k=0;k<parent.w().nx();k++){
                    for(size_t m=0;m<parent.w().nz();m++){
                        b_subsystem_vec_sample_addends.push_back(l_env_sample[parent.order()][s].at(k)+u_env_sample[parent.order()][s].at(m)+parent.w().at(k,l,m));
                    }
                }
                b_subsystem_vec_sample[s].at(l)=lse(b_subsystem_vec_sample_addends);
            }
        }
    }
    std::vector<double> z_addends;
    for(size_t k=0;k<current.w().nz();k++){
        z_addends.push_back(a_subsystem_vec_z.at(k)+b_subsystem_vec_z.at(k));
    }
    double z=lse(z_addends);
    double s_a=0;
    double s_b=0;
    double s_ab=0;
    #pragma omp parallel for reduction(-:s_a,s_b,s_ab)
    for(size_t s=0;s<n_samples;s++){
        std::vector<double> s_a_addends;
        std::vector<double> s_b_addends;
        std::vector<double> s_ab_addends;
        for(size_t k=0;k<current.w().nz();k++){
            s_a_addends.push_back(a_subsystem_vec_sample[s].at(k)+b_subsystem_vec_z.at(k));
            s_b_addends.push_back(a_subsystem_vec_z.at(k)+b_subsystem_vec_sample[s].at(k));
            s_ab_addends.push_back(a_subsystem_vec_sample[s].at(k)+b_subsystem_vec_sample[s].at(k));
        }
        s_a-=lse(s_a_addends);
        s_b-=lse(s_b_addends);
        s_ab-=lse(s_ab_addends);
    }
    double bmi=((s_a+s_b-s_ab)/(double) n_samples)+z;
    // std::cout<<"current:\n"<<(std::string)current.w()<<"\n";
    // std::cout<<"parent:\n"<<(std::string)parent.w()<<"\n";
    // std::cout<<"z: "<<z<<"\n";
    // std::cout<<"s_a: "<<s_a<<"\n";
    // std::cout<<"s_b: "<<s_b<<"\n";
    // std::cout<<"s_ab: "<<s_ab<<"\n";
    // std::cout<<"("<<current.order()<<","<<parent.order()<<") bmi: "<<bmi<<"\n";
    return bmi;
}

double optimize::calc_bmi_input(size_t idx,size_t input_rank,std::vector<sample_data>& samples,bond& parent,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample){
    if(input_rank==1){return 0;} //bmi is nonnegative and upper-bounded by bond dimension, so dim=1 -> bmi=0
    // size_t top_idx=
    size_t n_samples=samples.size();
    array1d<double> a_subsystem_vec_z(input_rank);
    array1d<double> b_subsystem_vec_z(input_rank);
    std::vector<array1d<double> > a_subsystem_vec_sample(n_samples);
    std::vector<array1d<double> > b_subsystem_vec_sample(n_samples);
    #pragma omp parallel for
    for(size_t s=0;s<n_samples;s++){
        array1d<double> vec_e(input_rank);
        for(size_t a=0;a<vec_e.nx();a++){
            if(a!=(samples[s].s()[idx]-1)){ //if a==samples[s].s()[n]-1, element is log(1)=0. else log(0)=-inf
                vec_e.at(a)=log(1e-100);
            }
        }
        a_subsystem_vec_sample[s]=vec_e;
        b_subsystem_vec_sample[s]=array1d<double>(input_rank);
    }
    if((idx==parent.v1())){
        for(size_t k=0;k<parent.w().nx();k++){
        // for(size_t k=0;k<input_rank;k++){
            std::vector<double> b_subsystem_vec_z_addends;
            for(size_t l=0;l<parent.w().ny();l++){
                for(size_t m=0;m<parent.w().nz();m++){
            // for(size_t l=0;l<r_env_z[parent.order()].nx();l++){
                // for(size_t m=0;m<u_env_z[parent.order()].nx();m++){
                    b_subsystem_vec_z_addends.push_back(r_env_z[parent.order()].at(l)+u_env_z[parent.order()].at(m)+parent.w().at(k,l,m));
                }
            }
            b_subsystem_vec_z.at(k)=lse(b_subsystem_vec_z_addends);
        }
        #pragma omp parallel for
        for(size_t s=0;s<n_samples;s++){
            for(size_t k=0;k<parent.w().nx();k++){
            // for(size_t k=0;k<input_rank;k++){
                std::vector<double> b_subsystem_vec_sample_addends;
                for(size_t l=0;l<parent.w().ny();l++){
                    for(size_t m=0;m<parent.w().nz();m++){
                // for(size_t l=0;l<r_env_z[parent.order()].nx();l++){
                    // for(size_t m=0;m<u_env_z[parent.order()].nx();m++){
                        b_subsystem_vec_sample_addends.push_back(r_env_sample[parent.order()][s].at(l)+u_env_sample[parent.order()][s].at(m)+parent.w().at(k,l,m));
                    }
                }
                b_subsystem_vec_sample[s].at(k)=lse(b_subsystem_vec_sample_addends);
            }
        }
    }
    else{
        for(size_t l=0;l<parent.w().ny();l++){
        // for(size_t l=0;l<input_rank;l++){
            std::vector<double> b_subsystem_vec_z_addends;
            for(size_t k=0;k<parent.w().nx();k++){
                for(size_t m=0;m<parent.w().nz();m++){
            // for(size_t k=0;k<l_env_z[parent.order()].nx();k++){
                // for(size_t m=0;m<u_env_z[parent.order()].nx();m++){
                    b_subsystem_vec_z_addends.push_back(l_env_z[parent.order()].at(k)+u_env_z[parent.order()].at(m)+parent.w().at(k,l,m));
                }
            }
            b_subsystem_vec_z.at(l)=lse(b_subsystem_vec_z_addends);
        }
        #pragma omp parallel for
        for(size_t s=0;s<n_samples;s++){
            for(size_t l=0;l<parent.w().ny();l++){
            // for(size_t l=0;l<input_rank;l++){
                std::vector<double> b_subsystem_vec_sample_addends;
                for(size_t k=0;k<parent.w().nx();k++){
                    for(size_t m=0;m<parent.w().nz();m++){
                // for(size_t k=0;k<l_env_z[parent.order()].nx();k++){
                    // for(size_t m=0;m<u_env_z[parent.order()].nx();m++){
                        b_subsystem_vec_sample_addends.push_back(l_env_sample[parent.order()][s].at(k)+u_env_sample[parent.order()][s].at(m)+parent.w().at(k,l,m));
                    }
                }
                b_subsystem_vec_sample[s].at(l)=lse(b_subsystem_vec_sample_addends);
            }
        }
    }
    std::vector<double> z_addends;
    for(size_t k=0;k<input_rank;k++){
        z_addends.push_back(a_subsystem_vec_z.at(k)+b_subsystem_vec_z.at(k));
    }
    double z=lse(z_addends);
    double s_a=0;
    double s_b=0;
    double s_ab=0;
    #pragma omp parallel for reduction(-:s_a,s_b,s_ab)
    for(size_t s=0;s<n_samples;s++){
        std::vector<double> s_a_addends;
        std::vector<double> s_b_addends;
        std::vector<double> s_ab_addends;
        for(size_t k=0;k<input_rank;k++){
            s_a_addends.push_back(a_subsystem_vec_sample[s].at(k)+b_subsystem_vec_z.at(k));
            s_b_addends.push_back(a_subsystem_vec_z.at(k)+b_subsystem_vec_sample[s].at(k));
            s_ab_addends.push_back(a_subsystem_vec_sample[s].at(k)+b_subsystem_vec_sample[s].at(k));
        }
        s_a-=lse(s_a_addends);
        s_b-=lse(s_b_addends);
        s_ab-=lse(s_ab_addends);
    }
    double bmi=((s_a+s_b-s_ab)/(double) n_samples)+z;
    // std::cout<<"z: "<<z<<"\n";
    // std::cout<<"s_a: "<<s_a<<"\n";
    // std::cout<<"s_b: "<<s_b<<"\n";
    // std::cout<<"s_ab: "<<s_ab<<"\n";
    // std::cout<<"("<<idx<<","<<parent.order()<<") bmi: "<<bmi<<"\n";
    return bmi;
}

size_t inner_nmf(array3d<double>& fused_mat,array3d<double>& mat1,array3d<double>& mat2,size_t r_max,bool compress_r){
    size_t r;
    if(compress_r){
        r=1;
        size_t upper_bound_r_max=(fused_mat.nx()<fused_mat.ny())?fused_mat.nx():fused_mat.ny();
        while(r<=((upper_bound_r_max<r_max)?upper_bound_r_max:r_max)){
            mat1=array3d<double>(fused_mat.nx(),r,1);
            mat2=array3d<double>(r,fused_mat.ny(),1);
            double recon_err=nmf(fused_mat,mat1,mat2,r); //nmf factors stored in mat1,mat2
            // std::cout<<r<<" "<<recon_err<<"\n";
            if(recon_err<1e-12){break;}
            if(r==((upper_bound_r_max<r_max)?upper_bound_r_max:r_max)){break;}
            r++;
        }
    }
    else{
        r=(fused_mat.nx()<fused_mat.ny())?fused_mat.nx():fused_mat.ny(); //max rank is min(row rank, col rank)
        r=(r<r_max)?r:r_max;
        mat1=array3d<double>(fused_mat.nx(),r,1);
        mat2=array3d<double>(r,fused_mat.ny(),1);
        double recon_err=nmf(fused_mat,mat1,mat2,r); //nmf factors stored in mat1,mat2
    }
    return r;
}

template<typename cmp>
void inner_updates(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,std::set<size_t>& done_idxs,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
    w=calc_w(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
    
    //update l/r env of parent and u_env of current
    // aux_update_lr_cache(current,parent,l_env_z,r_env_z,l_env_sample,r_env_sample);
    // aux_update_u_cache(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    
    // z=update_cache_z(g,current.order(),l_env_z,r_env_z,u_env_z,done_idxs);
    // w=update_cache_w(g,current.order(),l_env_sample,r_env_sample,u_env_sample,done_idxs);
    
    array3d<double> current_m(current.w().nx(),current.w().ny(),current.w().nz());
    array3d<double> current_v(current.w().nx(),current.w().ny(),current.w().nz());
    array3d<double> parent_m(parent.w().nx(),parent.w().ny(),parent.w().nz());
    array3d<double> parent_v(parent.w().nx(),parent.w().ny(),parent.w().nz());
    
    //single-site updates
    for(size_t t2=1;t2<=single_site_update_count;t2++){
        optimize::site_update(current,z,w,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample,current_m,current_v,t2,lr,beta1,beta2,epsilon);
    
        g.es().erase(it);
        it=g.es().insert(current);
        
        // z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
        // w=calc_w(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        
        // update l/r env of parent
        aux_update_lr_cache(current,parent,l_env_z,r_env_z,l_env_sample,r_env_sample);
        aux_update_u_cache(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);

        z=update_cache_z(g,current.order(),l_env_z,r_env_z,u_env_z,done_idxs);
        w=update_cache_w(g,current.order(),l_env_sample,r_env_sample,u_env_sample,done_idxs);
        
        optimize::site_update(parent,z,w,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample,parent_m,parent_v,t2,lr,beta1,beta2,epsilon);
        
        g.es().erase(it_parent);
        it_parent=g.es().insert(parent);
        
        // z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
        // w=calc_w(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        
        //update u_env of current
        aux_update_lr_cache(current,parent,l_env_z,r_env_z,l_env_sample,r_env_sample);
        aux_update_u_cache(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
        
        z=update_cache_z(g,parent.order(),l_env_z,r_env_z,u_env_z,done_idxs);
        w=update_cache_w(g,parent.order(),l_env_sample,r_env_sample,u_env_sample,done_idxs);
    }
}
template void inner_updates(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::set<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);

double inner_bmi(bond& current,bond& parent,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample){
    //calculate bmi using improved tensors
    // double bmi1=fabs(optimize::calc_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample)); //take abs
    double bmi=optimize::calc_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample); //take abs
    // if(bmi<-log(current.w().nz())){bmi=2*log(current.w().nz());}
    // if(bmi<-1e-8){bmi=2*log(current.w().nz());}
    if(bmi<-1e-4){bmi=std::numeric_limits<double>::quiet_NaN();}
    return bmi;
}

template<typename cmp>
double way1(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,std::set<size_t>& done_idxs,array4d<double>& fused,size_t r_max,bool compress_r,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    //split into two 3-leg tensors again via NMF: way 1
    array3d<double> fused_mat(fused.nx()*fused.ny(),fused.nz()*fused.nw(),1);
    for(size_t i=0;i<fused.nx();i++){
        for(size_t j=0;j<fused.ny();j++){
            for(size_t k=0;k<fused.nz();k++){
                for(size_t l=0;l<fused.nw();l++){
                    fused_mat.at((fused.ny()*i)+j,(fused.nw()*k)+l,0)=fused.at(i,j,k,l); //(ij)(kl) pairing
                }
            }
        }
    }
    
    array3d<double> mat1;
    array3d<double> mat2;
    size_t r=inner_nmf(fused_mat,mat1,mat2,r_max,compress_r);
    
    current.w()=array3d<double>(fused.nx(),fused.ny(),r);
    parent.w()=(parent.v1()==current.order())?array3d<double>(r,fused.nz(),fused.nw()):array3d<double>(fused.nz(),r,fused.nw());
    for(size_t i=0;i<mat1.nx();i++){
        for(size_t j=0;j<mat1.ny();j++){
            double val=mat1.at(i,j,0);
            current.w().at(i/current.w().ny(),i%current.w().ny(),j)=log((val>epsilon)?val:epsilon);
        }
    }
    for(size_t i=0;i<mat2.nx();i++){
        for(size_t j=0;j<mat2.ny();j++){
            double val;
            if(parent.v1()==current.order()){
                val=mat2.at(i,j,0);
                parent.w().at(i,j/parent.w().nz(),j%parent.w().nz())=log((val>epsilon)?val:epsilon);
            }
            else{
                val=mat2.at(i,j,0);
                parent.w().at(j/parent.w().nz(),i,j%parent.w().nz())=log((val>epsilon)?val:epsilon);
            }
        }
    }
    
    g.vs()[current.order()].rank()=r;
    
    g.es().erase(it);
    it=g.es().insert(current);
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    
    inner_updates(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,done_idxs,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    double bmi=inner_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way1: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    g.vs()[current.order()].p_bond()=current;
    g.vs()[parent.order()].p_bond()=parent;
    
    g.es().erase(it);
    it=g.es().insert(current);
    return bmi;
}
template double way1(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::set<size_t>&,array4d<double>&,size_t,bool,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);

template<typename cmp>
double way2(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,std::set<size_t>& done_idxs,array4d<double>& fused,size_t r_max,bool compress_r,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    //split into two 3-leg tensors again via NMF: way 2
    array3d<double> fused_mat(fused.nx()*fused.nz(),fused.ny()*fused.nw(),1);
    for(size_t i=0;i<fused.nx();i++){
        for(size_t j=0;j<fused.nz();j++){
            for(size_t k=0;k<fused.ny();k++){
                for(size_t l=0;l<fused.nw();l++){
                    fused_mat.at((fused.nz()*i)+j,(fused.nw()*k)+l,0)=fused.at(i,k,j,l); //(ik)(jl) pairing
                }
            }
        }
    }
    
    array3d<double> mat1;
    array3d<double> mat2;
    size_t r=inner_nmf(fused_mat,mat1,mat2,r_max,compress_r);
    
    //fix neighbors
    size_t swap;
    if(current.order()==parent.v1()){
        swap=current.v2();
        current.v2()=parent.v2();
        parent.v2()=swap;
    }
    else{
        swap=current.v2();
        current.v2()=parent.v1();
        parent.v1()=swap;
    }
    
    g.vs()[current.v1()].u_idx()=current.order();
    g.vs()[current.v2()].u_idx()=current.order();
    // if(current.v1()>current.v2()){
        // swap=current.v1();
        // current.v1()=current.v2();
        // current.v2()=swap;
    // }
    g.vs()[current.order()].l_idx()=current.v1();
    g.vs()[current.order()].r_idx()=current.v2();
    g.vs()[parent.v1()].u_idx()=parent.order();
    g.vs()[parent.v2()].u_idx()=parent.order();
    // if(parent.v1()>parent.v2()){
        // swap=parent.v1();
        // parent.v1()=parent.v2();
        // parent.v2()=swap;
    // }
    g.vs()[parent.order()].l_idx()=parent.v1();
    g.vs()[parent.order()].r_idx()=parent.v2();
    
    current.w()=array3d<double>(fused.nx(),fused.nz(),r);
    parent.w()=(parent.v1()==current.order())?array3d<double>(r,fused.ny(),fused.nw()):array3d<double>(fused.ny(),r,fused.nw());
    g.vs()[current.order()].rank()=r;
    for(size_t i=0;i<mat1.nx();i++){
        for(size_t j=0;j<mat1.ny();j++){
            double val=mat1.at(i,j,0);
            current.w().at(i/current.w().ny(),i%current.w().ny(),j)=log((val>epsilon)?val:epsilon);
        }
    }
    for(size_t i=0;i<mat2.nx();i++){
        for(size_t j=0;j<mat2.ny();j++){
            double val;
            val=mat2.at(i,j,0);
            if(parent.v1()==current.order()){
                parent.w().at(i,j/parent.w().nz(),j%parent.w().nz())=log((val>epsilon)?val:epsilon);
            }
            else{
                parent.w().at(j/parent.w().nz(),i,j%parent.w().nz())=log((val>epsilon)?val:epsilon);
            }
        }
    }
    if((current.w().nx()!=g.vs()[current.v1()].rank())||(current.w().ny()!=g.vs()[current.v2()].rank())||(current.w().nz()!=g.vs()[current.order()].rank())){
        std::cout<<"mismatch in current: ("<<current.w().nx()<<","<<current.w().ny()<<","<<current.w().nz()<<") vs ("<<g.vs()[current.v1()].rank()<<","<<g.vs()[current.v2()].rank()<<","<<g.vs()[current.order()].rank()<<")\n";
        exit(1);
    }
    if((parent.w().nx()!=g.vs()[parent.v1()].rank())||(parent.w().ny()!=g.vs()[parent.v2()].rank())||(parent.w().nz()!=g.vs()[parent.order()].rank())){
        std::cout<<"mismatch in parent: ("<<parent.w().nx()<<","<<parent.w().ny()<<","<<parent.w().nz()<<") vs ("<<g.vs()[parent.v1()].rank()<<","<<g.vs()[parent.v2()].rank()<<","<<g.vs()[parent.order()].rank()<<")\n";
        exit(1);
    }
    
    //fix depths
    current.depth()=((g.vs()[current.v1()].depth()>g.vs()[current.v2()].depth())?g.vs()[current.v1()].depth():g.vs()[current.v2()].depth())+1;
    g.vs()[current.order()].depth()=current.depth();
    parent.depth()=((g.vs()[parent.v1()].depth()>g.vs()[parent.v2()].depth())?g.vs()[parent.v1()].depth():g.vs()[parent.v2()].depth())+1;
    g.vs()[parent.order()].depth()=parent.depth();
    
    g.es().erase(it);
    it=g.es().insert(current);
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    
    auto d_it=it;
    // std::cout<<(std::string) g<<"\n";
    while(1){
        bond b=*d_it;
        b.depth()=((g.vs()[b.v1()].depth()>g.vs()[b.v2()].depth())?g.vs()[b.v1()].depth():g.vs()[b.v2()].depth())+1;
        g.vs()[b.order()].depth()=b.depth();
        if(b.depth()!=(*d_it).depth()){
            auto new_d_it=g.es().insert(b);
            g.es().erase(d_it);
            d_it=new_d_it;
        }
        if((*d_it).order()==g.vs().size()-1){
            break;
        }
        bond key;
        key.todo()=0;
        key.order()=g.vs()[(*d_it).order()].u_idx();
        key.depth()=g.vs()[key.order()].depth();
        key.bmi()=-1e50;
        key.virt_count()=2;
        d_it=g.es().lower_bound(key);
    }
    // std::cout<<(std::string) g<<"\n";
    
    inner_updates(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,done_idxs,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    double bmi=inner_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way2: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    g.vs()[current.order()].p_bond()=current;
    g.vs()[parent.order()].p_bond()=parent;
    
    g.es().erase(it);
    it=g.es().insert(current);
    return bmi;
}
template double way2(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::set<size_t>&,array4d<double>&,size_t,bool,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);

template<typename cmp>
double way3(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,std::set<size_t>& done_idxs,array4d<double>& fused,size_t r_max,bool compress_r,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    //split into two 3-leg tensors again via NMF: way 2
    array3d<double> fused_mat(fused.nz()*fused.ny(),fused.nx()*fused.nw(),1);
    for(size_t i=0;i<fused.nz();i++){
        for(size_t j=0;j<fused.ny();j++){
            for(size_t k=0;k<fused.nx();k++){
                for(size_t l=0;l<fused.nw();l++){
                    fused_mat.at((fused.ny()*i)+j,(fused.nw()*k)+l,0)=fused.at(k,j,i,l); //(kj)(il) pairing
                }
            }
        }
    }
    
    array3d<double> mat1;
    array3d<double> mat2;
    size_t r=inner_nmf(fused_mat,mat1,mat2,r_max,compress_r);
    
    //fix neighbors
    size_t swap;
    if(current.order()==parent.v2()){
        swap=current.v1();
        current.v1()=parent.v1();
        parent.v1()=swap;
    }
    else{
        swap=current.v1();
        current.v1()=parent.v2();
        parent.v2()=swap;
    }
    
    g.vs()[current.v1()].u_idx()=current.order();
    g.vs()[current.v2()].u_idx()=current.order();
    // if(current.v1()>current.v2()){
        // swap=current.v1();
        // current.v1()=current.v2();
        // current.v2()=swap;
    // }
    g.vs()[current.order()].l_idx()=current.v1();
    g.vs()[current.order()].r_idx()=current.v2();
    g.vs()[parent.v1()].u_idx()=parent.order();
    g.vs()[parent.v2()].u_idx()=parent.order();
    // if(parent.v1()>parent.v2()){
        // swap=parent.v1();
        // parent.v1()=parent.v2();
        // parent.v2()=swap;
    // }
    g.vs()[parent.order()].l_idx()=parent.v1();
    g.vs()[parent.order()].r_idx()=parent.v2();
    
    current.w()=array3d<double>(fused.nz(),fused.ny(),r);
    parent.w()=(parent.v1()==current.order())?array3d<double>(r,fused.nx(),fused.nw()):array3d<double>(fused.nx(),r,fused.nw());
    g.vs()[current.order()].rank()=r;
    for(size_t i=0;i<mat1.nx();i++){
        for(size_t j=0;j<mat1.ny();j++){
            double val=mat1.at(i,j,0);
            current.w().at(i/current.w().ny(),i%current.w().ny(),j)=log((val>epsilon)?val:epsilon);
        }
    }
    for(size_t i=0;i<mat2.nx();i++){
        for(size_t j=0;j<mat2.ny();j++){
            double val;
            val=mat2.at(i,j,0);
            if(parent.v1()==current.order()){
                parent.w().at(i,j/parent.w().nz(),j%parent.w().nz())=log((val>epsilon)?val:epsilon);
            }
            else{
                parent.w().at(j/parent.w().nz(),i,j%parent.w().nz())=log((val>epsilon)?val:epsilon);
            }
        }
    }
    if((current.w().nx()!=g.vs()[current.v1()].rank())||(current.w().ny()!=g.vs()[current.v2()].rank())||(current.w().nz()!=g.vs()[current.order()].rank())){
        std::cout<<"mismatch in current: ("<<current.w().nx()<<","<<current.w().ny()<<","<<current.w().nz()<<") vs ("<<g.vs()[current.v1()].rank()<<","<<g.vs()[current.v2()].rank()<<","<<g.vs()[current.order()].rank()<<")\n";
        exit(1);
    }
    if((parent.w().nx()!=g.vs()[parent.v1()].rank())||(parent.w().ny()!=g.vs()[parent.v2()].rank())||(parent.w().nz()!=g.vs()[parent.order()].rank())){
        std::cout<<"mismatch in parent: ("<<parent.w().nx()<<","<<parent.w().ny()<<","<<parent.w().nz()<<") vs ("<<g.vs()[parent.v1()].rank()<<","<<g.vs()[parent.v2()].rank()<<","<<g.vs()[parent.order()].rank()<<")\n";
        exit(1);
    }
    
    //fix depths
    current.depth()=((g.vs()[current.v1()].depth()>g.vs()[current.v2()].depth())?g.vs()[current.v1()].depth():g.vs()[current.v2()].depth())+1;
    g.vs()[current.order()].depth()=current.depth();
    parent.depth()=((g.vs()[parent.v1()].depth()>g.vs()[parent.v2()].depth())?g.vs()[parent.v1()].depth():g.vs()[parent.v2()].depth())+1;
    g.vs()[parent.order()].depth()=parent.depth();
    
    g.es().erase(it);
    it=g.es().insert(current);
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    
    auto d_it=it;
    while(1){
        bond b=*d_it;
        b.depth()=((g.vs()[b.v1()].depth()>g.vs()[b.v2()].depth())?g.vs()[b.v1()].depth():g.vs()[b.v2()].depth())+1;
        g.vs()[b.order()].depth()=b.depth();
        if(b.depth()!=(*d_it).depth()){
            auto new_d_it=g.es().insert(b);
            g.es().erase(d_it);
            d_it=new_d_it;
        }
        if((*d_it).order()==g.vs().size()-1){
            break;
        }
        bond key;
        key.todo()=0;
        key.order()=g.vs()[(*d_it).order()].u_idx();
        key.depth()=g.vs()[key.order()].depth();
        key.bmi()=-1e50;
        key.virt_count()=2;
        d_it=g.es().lower_bound(key);
    }
    // std::cout<<(std::string) g<<"\n";
    
    inner_updates(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,done_idxs,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    double bmi=inner_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way3: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    g.vs()[current.order()].p_bond()=current;
    g.vs()[parent.order()].p_bond()=parent;
    
    g.es().erase(it);
    it=g.es().insert(current);
    return bmi;
}
template double way3(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::set<size_t>&,array4d<double>&,size_t,bool,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);