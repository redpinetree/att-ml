#include <cstring>
#include <limits>
#include <list>
#include <random>

#include "omp.h"

#include "algorithm_nll.hpp"
#include "mat_ops.hpp"
#include "optimize_nll.hpp"
#include "ttn_ops.hpp"
#include "utils.hpp"

template<typename cmp>
double optimize::opt_struct_nll(graph<cmp>& g,std::vector<std::vector<array1d<double> > >& train_samples,std::vector<int>& train_labels,std::vector<std::vector<array1d<double> > >& test_samples,std::vector<int>& test_labels,int iter_max,int r_max,bool compress_r,double lr,int batch_size,std::map<int,double>& train_nll_history,std::map<int,double>& test_nll_history,std::map<int,int>& sweep_history,bool struct_opt){
    if(iter_max==0){return 0;}
    int single_site_update_count=10;
    double prev_nll=1e50;
    double nll=0;
    double test_nll=0;
    //adam variables
    // double lr=0.001;
    double beta1=0.9;
    double beta2=0.999;
    double epsilon=1e-16;
    //initialize adam m,v caches
    std::map<std::pair<int,int>,array4d<double> > fused_m;
    std::map<std::pair<int,int>,array4d<double> > fused_v;
    for(auto it=g.es().begin();it!=--g.es().end();++it){
        bond current=*it;
        bond parent=g.vs()[g.vs()[(*it).order()].u_idx()].p_bond();
        std::pair<int,int> key=(current.order()<parent.order())?std::make_pair(current.order(),parent.order()):std::make_pair(parent.order(),current.order());
        fused_m[key]=array4d<double>(current.w().nx(),current.w().ny(),(parent.v1()==current.order())?parent.w().ny():parent.w().nx(),parent.w().nz());
        fused_v[key]=array4d<double>(current.w().nx(),current.w().ny(),(parent.v1()==current.order())?parent.w().ny():parent.w().nx(),parent.w().nz());
    }
    //initialize minibatches
    int batch_start_idx=0;
    std::vector<int> batch_shuffle(train_samples.size());
    for(int i=0;i<train_samples.size();i++){
        batch_shuffle[i]=i;
    }
    std::vector<std::vector<array1d<double> > > train_samples_batch(train_samples.begin()+batch_start_idx,((batch_start_idx+batch_size<train_samples.size())?(train_samples.begin()+batch_start_idx+batch_size):train_samples.end()));
    std::vector<int> train_labels_batch;
    if(train_labels.size()!=0){
        std::vector<int> train_labels_batch(train_labels.begin()+batch_start_idx,((batch_start_idx+batch_size<train_labels.size())?(train_labels.begin()+batch_start_idx+batch_size):train_labels.end()));
    }
    batch_start_idx=(batch_start_idx+batch_size<train_samples.size())?batch_start_idx+batch_size:0;
    
    std::vector<array1d<double> > l_env_z;
    l_env_z.reserve(g.vs().size());
    std::vector<array1d<double> > r_env_z;
    r_env_z.reserve(g.vs().size());
    std::vector<array1d<double> > u_env_z;
    u_env_z.reserve(g.vs().size());
    std::vector<array2d<double> > l_env_sample;
    l_env_sample.reserve(g.vs().size());
    std::vector<array2d<double> > r_env_sample;
    r_env_sample.reserve(g.vs().size());
    std::vector<array2d<double> > u_env_sample;
    u_env_sample.reserve(g.vs().size());
    double z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
    while(fabs(z-1)>=1e-8){ //normalize tensors so that z=1
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond current=*it;
            for(int i=0;i<current.w().nx();i++){
                for(int j=0;j<current.w().ny();j++){
                    for(int k=0;k<current.w().nz();k++){
                        current.w().at(i,j,k)/=pow(z,1/(double) g.es().size());
                    }
                }
            }
            g.vs()[current.order()].p_bond()=current;
            g.es().erase(it);
            it=g.es().insert(current);
        }
        z=calc_z(g,l_env_z,r_env_z,u_env_z);
    }
    std::vector<double> w=calc_w(g,train_samples_batch,train_labels_batch,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
    
    double best_nll=1e50;
    double best_test_nll=1e50;
    std::uniform_real_distribution<> unif_dist(1e-16,1.0);
    std::vector<site> best_vs;
    std::multiset<bond,bmi_comparator> best_es;
    int t=1;
    int iter=1;
    while(iter<iter_max){
        //all tensors have an upstream tensor except the top, so we can loop over all tensors except the top to go through all bonds
        // std::cout<<"iter "<<t<<" ";
        std::set<int> done_idxs; //set to store processed bond idxs
        // std::cout<<(std::string)g<<"\n";
        auto it=g.es().begin();
        while(done_idxs.size()<g.es().size()){
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
                it=g.es().lower_bound(key);
                continue;
            }
            bond key;
            key.todo()=0;
            key.order()=g.vs()[(*it).order()].u_idx();
            key.depth()=g.vs()[key.order()].depth();
            key.bmi()=-1e50;
            auto it_parent=g.es().lower_bound(key);
            bond parent=*it_parent; //must be a dereferenced pointer to the actual object, not a copy!
            
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
            std::vector<array2d<double> > orig_l_env_sample=l_env_sample;
            std::vector<array2d<double> > orig_r_env_sample=r_env_sample;
            std::vector<array2d<double> > orig_u_env_sample=u_env_sample;
            
            double bmi1=way1(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,fused,r_max,compress_r,train_samples_batch,train_labels_batch,single_site_update_count,lr,beta1,beta2,epsilon);
            
            if(struct_opt){
                graph<cmp> way1_g=g;
                bond way1_current=current;
                bond way1_parent=parent;
                double way1_z=z;
                std::vector<array1d<double> > way1_l_env_z=l_env_z;
                std::vector<array1d<double> > way1_r_env_z=r_env_z;
                std::vector<array1d<double> > way1_u_env_z=u_env_z;
                std::vector<double> way1_w=w;
                std::vector<array2d<double> > way1_l_env_sample=l_env_sample;
                std::vector<array2d<double> > way1_r_env_sample=r_env_sample;
                std::vector<array2d<double> > way1_u_env_sample=u_env_sample;
                
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
                
                double bmi2=way2(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,fused,r_max,compress_r,train_samples_batch,train_labels_batch,single_site_update_count,lr,beta1,beta2,epsilon);
                
                graph<bmi_comparator> way2_g=g;
                bond way2_current=current;
                bond way2_parent=parent;
                double way2_z=z;
                std::vector<array1d<double> > way2_l_env_z=l_env_z;
                std::vector<array1d<double> > way2_r_env_z=r_env_z;
                std::vector<array1d<double> > way2_u_env_z=u_env_z;
                std::vector<double> way2_w=w;
                std::vector<array2d<double> > way2_l_env_sample=l_env_sample;
                std::vector<array2d<double> > way2_r_env_sample=r_env_sample;
                std::vector<array2d<double> > way2_u_env_sample=u_env_sample;
                
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
                
                double bmi3=way3(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,fused,r_max,compress_r,train_samples_batch,train_labels_batch,single_site_update_count,lr,beta1,beta2,epsilon);
                
                graph<bmi_comparator> way3_g=g;
                bond way3_current=current;
                bond way3_parent=parent;
                double way3_z=z;
                std::vector<array1d<double> > way3_l_env_z=l_env_z;
                std::vector<array1d<double> > way3_r_env_z=r_env_z;
                std::vector<array1d<double> > way3_u_env_z=u_env_z;
                std::vector<double> way3_w=w;
                std::vector<array2d<double> > way3_l_env_sample=l_env_sample;
                std::vector<array2d<double> > way3_r_env_sample=r_env_sample;
                std::vector<array2d<double> > way3_u_env_sample=u_env_sample;
                
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
                    it_parent=g.es().find(parent);
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
                    it_parent=g.es().find(parent);
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
                    it_parent=g.es().find(parent);
                    // std::cout<<"selected bmi3\n";
                }
            }
            
            done_idxs.insert(current.order());
            // std::cout<<(std::string)g<<"\n";
            
            normalize(current.w());
            normalize(parent.w());
            
            g.vs()[current.order()].p_bond()=current;
            g.es().erase(it);
            it=g.es().insert(current);
            g.vs()[parent.order()].p_bond()=parent;
            g.es().erase(it_parent);
            it_parent=g.es().insert(parent);
            
            z=calc_z(g,l_env_z,r_env_z,u_env_z);
            w=calc_w(g,train_samples_batch,train_labels_batch,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
            
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
            it=g.es().lower_bound(key);
            
            if(iter==iter_max){break;}
            iter++;
        }
        //calculate train nll
        nll=0;
        #pragma omp parallel for reduction(-:nll)
        for(int s=0;s<train_samples_batch.size();s++){
            nll-=log(w[s]);
        }
        nll/=(double) train_samples_batch.size();
        nll+=log(z);
        train_nll_history.insert(std::pair<int,double>(iter,nll));
        
        //calculate test nll per sweep
        if(test_samples.size()!=0){
            std::vector<double> test_w=calc_w(g,test_samples,test_labels);
            test_nll=0;
            #pragma omp parallel for reduction(-:test_nll)
            for(int s=0;s<test_samples.size();s++){
                test_nll-=log(test_w[s]);
            }
            test_nll/=(double) test_samples.size();
            test_nll+=log(z);
            test_nll_history.insert(std::pair<int,double>(iter,test_nll));
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
            train_nll_history.insert(std::pair<int,double>(iter,nll));
            break;
        }
        if(fabs((prev_nll-nll)/nll)<1e-6){
            std::cout<<"NLL optimization converged after "<<iter<<" iterations.\n";
            if(test_samples.size()!=0){
                std::cout<<"sweep "<<t<<" iter "<<iter<<" train nll="<<nll<<" test nll="<<test_nll<<"\n";
            }
            else{
                std::cout<<"sweep "<<t<<" iter "<<iter<<" train nll="<<nll<<"\n";
            }
            train_nll_history.insert(std::pair<int,double>(iter,nll));
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
        sweep_history.insert(std::pair<int,double>(t,iter));
        
        //update minibatch and recalculate minibatch-dependent w
        train_samples_batch=std::vector<std::vector<array1d<double> > >(train_samples.begin()+batch_start_idx,((batch_start_idx+batch_size<train_samples.size())?(train_samples.begin()+batch_start_idx+batch_size):train_samples.end()));
        if(train_labels.size()!=0){
            train_labels_batch=std::vector<int>(train_labels.begin()+batch_start_idx,((batch_start_idx+batch_size<train_labels.size())?(train_labels.begin()+batch_start_idx+batch_size):train_labels.end()));
        }
        batch_start_idx=(batch_start_idx+batch_size<train_samples.size())?batch_start_idx+batch_size:0;
        if(batch_start_idx==0){
            std::shuffle(std::begin(batch_shuffle),std::end(batch_shuffle),prng);
            std::vector<std::vector<array1d<double> > > train_samples_copy=train_samples;
            for(int i=0;i<train_samples.size();i++){
                train_samples[i]=train_samples_copy[batch_shuffle[i]];
            }
            if(train_labels.size()!=0){
                std::vector<int> train_labels_copy=train_labels;
                for(int i=0;i<train_labels.size();i++){
                    train_labels[i]=train_labels_copy[batch_shuffle[i]];
                }
            }
        }
        w=calc_w(g,train_samples_batch,train_labels_batch,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        t++;
    }
    //calculate final train nll
    // nll=0;
    // #pragma omp parallel for reduction(-:nll)
    // for(int s=0;s<train_samples.size();s++){
        // nll-=w[s]; //w[s] is log(w(s))
    // }
    // nll/=(double) train_samples.size();
    // nll+=z; //z is log(z)
    // std::cout<<"final nll="<<nll<<"\n";
    // train_nll_history.insert(std::pair<int,double>(iter,nll));
    // if(nll<best_nll){
        // best_nll=nll;
        // best_test_nll=test_nll;
        // best_vs=g.vs();
        // best_es=g.es();
    // }
    std::cout<<"best nll="<<best_nll<<"\n";
    if(test_samples.size()!=0){std::cout<<"best test nll="<<best_test_nll<<"\n";}
    t++;
    nll=best_nll;
    test_nll=best_test_nll;
    g.vs()=best_vs;
    g.es()=best_es;
    
    z=calc_z(g,l_env_z,r_env_z,u_env_z);
    w=calc_w(g,train_samples,train_labels,l_env_sample,r_env_sample,u_env_sample);
    
    return best_nll;
}
template double optimize::opt_struct_nll(graph<bmi_comparator>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,int,bool,double,int,std::map<int,double>&,std::map<int,double>&,std::map<int,int>&,bool);

template<typename cmp>
std::vector<int> optimize::classify(graph<cmp>& g,std::vector<std::vector<array1d<double> > >& samples,array2d<double>& probs){
    std::vector<array2d<double> > contracted_vectors; //batched vectors
    contracted_vectors.reserve(g.vs().size());
    int n_samples=samples.size();
    for(int n=0;n<g.vs().size();n++){
        int rank=g.vs()[n].rank();
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            array2d<double> vec(rank,n_samples);
            #pragma omp parallel for
            for(int s=0;s<n_samples;s++){
                memcpy(&(vec.e())[rank*s],&(samples[s][n].e())[0],rank*sizeof(double));
            }
            contracted_vectors.push_back(vec);
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(array2d<double>(0,n_samples));
        }
    }
    int contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()].nx()!=0)&&(contracted_vectors[(*it).v2()].nx()!=0)&&((it==--g.es().end())||(contracted_vectors[(*it).order()].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                array2d<double> res_vec((*it).w().nz(),n_samples);
                #pragma omp parallel for collapse(2)
                for(int s=0;s<n_samples;s++){
                    for(int k=0;k<(*it).w().nz();k++){
                        for(int i=0;i<(*it).w().nx();i++){
                            for(int j=0;j<(*it).w().ny();j++){
                                res_vec.at(k,s)+=contracted_vectors[(*it).v1()].at(i,s)*contracted_vectors[(*it).v2()].at(j,s)*(*it).w().at(i,j,k);
                            }
                        }
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
    for(int s=0;s<n_samples;s++){
        double prob_sum=0;
        for(int i=0;i<probs.nx();i++){
            probs.at(i,s)*=probs.at(i,s);
            prob_sum+=probs.at(i,s);
        }
        for(int i=0;i<probs.nx();i++){
            probs.at(i,s)/=prob_sum;
        }
    }
    
    std::vector<int> classes(n_samples);
    #pragma omp parallel for
    for(int s=0;s<n_samples;s++){
        for(int i=0;i<probs.nx();i++){
            if(probs.at(i,s)>probs.at(classes[s],s)){
                classes[s]=i;
            }
        }
    }
    return classes;
}
template std::vector<int> optimize::classify(graph<bmi_comparator>&,std::vector<std::vector<array1d<double> > >&,array2d<double>&);

void optimize::site_update(bond& b,double z,std::vector<double>& w,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample,array3d<double>& m_cache,array3d<double>& v_cache,int t,double lr,double beta1,double beta2,double epsilon){
    int b_order=b.order();
    array3d<double> b_w=b.w();
    array3d<double> dz(b_w.nx(),b_w.ny(),b_w.nz());
    #pragma omp parallel for collapse(3)
    for(int i=0;i<dz.nx();i++){
        for(int j=0;j<dz.ny();j++){
            for(int k=0;k<dz.nz();k++){
                dz.at(i,j,k)=l_env_z[b_order].at(i)*r_env_z[b_order].at(j)*u_env_z[b_order].at(k);
            }
        }
    }
    
    array4d<double> dw(b_w.nx(),b_w.ny(),b_w.nz(),w.size());
    #pragma omp parallel for collapse(4)
    for(int s=0;s<w.size();s++){
        for(int i=0;i<dw.nx();i++){
            for(int j=0;j<dw.ny();j++){
                for(int k=0;k<dw.nz();k++){
                    dw.at(i,j,k,s)=l_env_sample[b_order].at(i,s)*r_env_sample[b_order].at(j,s)*u_env_sample[b_order].at(k,s);
                }
            }
        }
    }
    
    array3d<double> grad(b_w.nx(),b_w.ny(),b_w.nz());
    array3d<double> grad_z_term(b_w.nx(),b_w.ny(),b_w.nz());
    array3d<double> grad_w_term(b_w.nx(),b_w.ny(),b_w.nz());
    #pragma omp parallel for collapse(3)
    for(int i=0;i<grad.nx();i++){
        for(int j=0;j<grad.ny();j++){
            for(int k=0;k<grad.nz();k++){
                grad_z_term.at(i,j,k)=dz.at(i,j,k)/z;
                double grad_w_term_sum=0;;
                for(int s=0;s<w.size();s++){
                    grad_w_term_sum+=dw.at(i,j,k,s)/w[s];
                }
                grad_w_term.at(i,j,k)=grad_w_term_sum/(double) w.size();
                //perform unconstrained gd on original problem
                grad.at(i,j,k)=grad_z_term.at(i,j,k)-grad_w_term.at(i,j,k);
            }
        }
    }
    double grad_norm=0;
    // #pragma omp parallel for reduction(+:grad_norm)
    for(int n=0;n<grad.nx()*grad.ny()*grad.nz();n++){
        int k=n%grad.nz();
        int j=(n/grad.nz())%grad.ny();
        int i=n/(grad.ny()*grad.nz());
        grad_norm+=grad.at(i,j,k)*grad.at(i,j,k);
    }
    grad_norm=sqrt(grad_norm);
    if(grad_norm>1){ //gradient clipping by norm
        #pragma omp parallel for collapse(3)
        for(int i=0;i<grad.nx();i++){
            for(int j=0;j<grad.ny();j++){
                for(int k=0;k<grad.nz();k++){
                    grad.at(i,j,k)=1*(grad.at(i,j,k)/grad_norm);
                }
            }
        }
    }
    #pragma omp parallel for collapse(3)
    for(int i=0;i<grad.nx();i++){
        for(int j=0;j<grad.ny();j++){
            for(int k=0;k<grad.nz();k++){
                double proj_grad=(b_w.at(i,j,k)>grad.at(i,j,k))?grad.at(i,j,k):b_w.at(i,j,k); //projected gradient
                m_cache.at(i,j,k)=(beta1*m_cache.at(i,j,k))+((1-beta1)*proj_grad);
                v_cache.at(i,j,k)=(beta2*v_cache.at(i,j,k))+((1-beta2)*proj_grad*proj_grad);
                double corrected_m=m_cache.at(i,j,k)/(1-pow(beta1,(double) t));
                double corrected_v=v_cache.at(i,j,k)/(1-pow(beta2,(double) t));
                b_w.at(i,j,k)=b_w.at(i,j,k)-(lr*0.01*b_w.at(i,j,k)); //weight decay (adamw)
                b_w.at(i,j,k)=b_w.at(i,j,k)-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                // b_w.at(i,j,k)=b_w.at(i,j,k)-(lr*proj_grad);
                if(b_w.at(i,j,k)<=0){b_w.at(i,j,k)=1e-16;}
                // if(b_w.at(i,j,k)>1){b_w.at(i,j,k)=1;}
            }
        }
    }
    b.w()=b_w;
    normalize(b.w());
    // normalize_using_z(b.w(),z);
}

array4d<double> optimize::fused_update(bond& b1,bond& b2,double z,std::vector<double>& w,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample,std::map<std::pair<int,int>,array4d<double> >& m_cache,std::map<std::pair<int,int>,array4d<double> >& v_cache,int t,double lr,double beta1,double beta2,double epsilon){
    int b1_order=b1.order();
    int b2_order=b2.order();
    int b2_v1=b2.v1();
    array3d<double> b1_w=b1.w();
    array3d<double> b2_w=b2.w();
    //calculate fused, dz and dw
    array4d<double> fused(b1_w.nx(),b1_w.ny(),(b2_v1==b1_order)?b2_w.ny():b2_w.nx(),b2_w.nz());
    #pragma omp parallel for collapse(4)
    for(int i=0;i<fused.nx();i++){
        for(int j=0;j<fused.ny();j++){
            for(int k=0;k<fused.nz();k++){
                for(int l=0;l<fused.nw();l++){
                    for(int m=0;m<b1_w.nz();m++){
                        fused.at(i,j,k,l)+=b1_w.at(i,j,m)*((b2_v1==b1_order)?b2_w.at(m,k,l):b2_w.at(k,m,l));
                    }
                }
            }
        }
    }
    
    array4d<double> dz(fused.nx(),fused.ny(),fused.nz(),fused.nw());
    #pragma omp parallel for collapse(4)
    for(int i=0;i<dz.nx();i++){
        for(int j=0;j<dz.ny();j++){
            for(int k=0;k<dz.nz();k++){
                for(int l=0;l<dz.nw();l++){
                    dz.at(i,j,k,l)=l_env_z[b1_order].at(i)*r_env_z[b1_order].at(j)*((b2_v1==b1_order)?r_env_z[b2_order].at(k):l_env_z[b2_order].at(k))*u_env_z[b2_order].at(l);
                }
            }
        }
    }
    array5d<double> dw(fused.nx(),fused.ny(),fused.nz(),fused.nw(),w.size());
    #pragma omp parallel for collapse(5)
    for(int s=0;s<w.size();s++){
        for(int i=0;i<dw.nx();i++){
            for(int j=0;j<dw.ny();j++){
                for(int k=0;k<dw.nz();k++){
                    for(int l=0;l<dw.nw();l++){
                        dw.at(i,j,k,l,s)=l_env_sample[b1_order].at(i,s)*r_env_sample[b1_order].at(j,s)*((b2_v1==b1_order)?r_env_sample[b2_order].at(k,s):l_env_sample[b2_order].at(k,s))*u_env_sample[b2_order].at(l,s);
                    }
                }
            }
        }
    }
    
    std::pair<int,int> key=(b1_order<b2_order)?std::make_pair(b1_order,b2_order):std::make_pair(b2_order,b1_order);
    
    //reset m and v caches if there is a change in size
    if((m_cache[key].nx()!=fused.nx())||(m_cache[key].ny()!=fused.ny())||(m_cache[key].nz()!=fused.nz())||(m_cache[key].nw()!=fused.nw())){
        m_cache[key]=array4d<double>(b1_w.nx(),b1_w.ny(),(b2_v1==b1_order)?b2_w.ny():b2_w.nx(),b2_w.nz());
        v_cache[key]=array4d<double>(b1_w.nx(),b1_w.ny(),(b2_v1==b1_order)?b2_w.ny():b2_w.nx(),b2_w.nz());
    }
    
    array4d<double> grad_2site(fused.nx(),fused.ny(),fused.nz(),fused.nw());
    array4d<double> grad_z_term_2site(grad_2site.nx(),grad_2site.ny(),grad_2site.nz(),grad_2site.nw());
    array4d<double> grad_w_term_2site(grad_2site.nx(),grad_2site.ny(),grad_2site.nz(),grad_2site.nw());
                    
    // std::cout<<"n="<<n<<"\n";
    // std::cout<<(std::string) b1_w<<"\n";
    #pragma omp parallel for collapse(4)
    for(int i=0;i<grad_2site.nx();i++){
        for(int j=0;j<grad_2site.ny();j++){
            for(int k=0;k<grad_2site.nz();k++){
                for(int l=0;l<grad_2site.nw();l++){
                    grad_z_term_2site.at(i,j,k,l)=dz.at(i,j,k,l)/z;
                    double grad_w_term_2site_sum=0;
                    for(int s=0;s<w.size();s++){
                        grad_w_term_2site_sum+=dw.at(i,j,k,l,s)/w[s];
                    }
                    grad_w_term_2site.at(i,j,k,l)=grad_w_term_2site_sum/(double) w.size();
                    //perform projected nonnegative gd on original problem
                    grad_2site.at(i,j,k,l)=grad_z_term_2site.at(i,j,k,l)-grad_w_term_2site.at(i,j,k,l);
                }
            }
        }
    }
    double grad_2site_norm=0;
    // #pragma omp parallel for reduction(+:grad_2site_norm)
    for(int n=0;n<grad_2site.nx()*grad_2site.ny()*grad_2site.nz()*grad_2site.nw();n++){
        int l=n%grad_2site.nw();
        int k=(n/grad_2site.nw())%grad_2site.nz();
        int j=(n/(grad_2site.nz()*grad_2site.nw()))%grad_2site.ny();
        int i=n/(grad_2site.ny()*grad_2site.nz()*grad_2site.nw());
        grad_2site_norm+=grad_2site.at(i,j,k,l)*grad_2site.at(i,j,k,l);
    }
    grad_2site_norm=sqrt(grad_2site_norm);
    if(grad_2site_norm>1){ //gradient clipping by norm
        #pragma omp parallel for collapse(4)
        for(int i=0;i<grad_2site.nx();i++){
            for(int j=0;j<grad_2site.ny();j++){
                for(int k=0;k<grad_2site.nz();k++){
                    for(int l=0;l<grad_2site.nw();l++){
                        grad_2site.at(i,j,k,l)=1*(grad_2site.at(i,j,k,l)/grad_2site_norm);
                    }
                }
            }
        }
    }
    #pragma omp parallel for collapse(4)
    for(int i=0;i<grad_2site.nx();i++){
        for(int j=0;j<grad_2site.ny();j++){
            for(int k=0;k<grad_2site.nz();k++){
                for(int l=0;l<grad_2site.nw();l++){
                    double proj_grad=(fused.at(i,j,k,l)>grad_2site.at(i,j,k,l))?grad_2site.at(i,j,k,l):fused.at(i,j,k,l); //projected gradient
                    m_cache[key].at(i,j,k,l)=(beta1*m_cache[key].at(i,j,k,l))+((1-beta1)*proj_grad);
                    v_cache[key].at(i,j,k,l)=(beta2*v_cache[key].at(i,j,k,l))+((1-beta2)*proj_grad*proj_grad);
                    double corrected_m=m_cache[key].at(i,j,k,l)/(1-pow(beta1,(double) t));
                    double corrected_v=v_cache[key].at(i,j,k,l)/(1-pow(beta2,(double) t));
                    fused.at(i,j,k,l)=fused.at(i,j,k,l)-(lr*0.01*fused.at(i,j,k,l)); //weight decay (adamw)
                    fused.at(i,j,k,l)=fused.at(i,j,k,l)-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                    // fused.at(i,j,k,l)=fused.at(i,j,k,l)-(lr*proj_grad);
                    if(fused.at(i,j,k,l)<=0){fused.at(i,j,k,l)=1e-16;}
                    // if(fused.at(i,j,k,l)>1){fused.at(i,j,k,l)=1;}
                }
            }
        }
    }
    normalize(fused);
    // normalize_using_z(fused,z);
    return fused;
}

double optimize::calc_bmi(bond& current,bond& parent,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample){
    int current_order=current.order();
    int parent_order=parent.order();
    array3d<double> current_w=current.w();
    array3d<double> parent_w=parent.w();
    if(current_w.nz()==1){return 0;} //bmi is nonnegative and upper-bounded by bond dimension, so dim=1 -> bmi=0
    int n_samples=u_env_sample[0].ny();
    array1d<double> a_subsystem_vec_z(current_w.nz());
    array1d<double> b_subsystem_vec_z(current_w.nz());
    array2d<double> a_subsystem_vec_sample(current_w.nz(),n_samples);
    array2d<double> b_subsystem_vec_sample(current_w.nz(),n_samples);
    #pragma omp parallel for
    for(int k=0;k<current_w.nz();k++){
        for(int i=0;i<current_w.nx();i++){
            for(int j=0;j<current_w.ny();j++){
                a_subsystem_vec_z.at(k)+=l_env_z[current_order].at(i)*r_env_z[current_order].at(j)*current_w.at(i,j,k);
            }
        }
    }
    #pragma omp parallel for collapse(2)
    for(int s=0;s<n_samples;s++){
        for(int k=0;k<current_w.nz();k++){
            for(int i=0;i<current_w.nx();i++){
                for(int j=0;j<current_w.ny();j++){
                    a_subsystem_vec_sample.at(k,s)+=l_env_sample[current_order].at(i,s)*r_env_sample[current_order].at(j,s)*current_w.at(i,j,k);
                }
            }
        }
    }
    if((current_order==parent.v1())){
        #pragma omp parallel for
        for(int k=0;k<parent_w.nx();k++){
            for(int l=0;l<parent_w.ny();l++){
                for(int m=0;m<parent_w.nz();m++){
                    b_subsystem_vec_z.at(k)+=r_env_z[parent_order].at(l)*u_env_z[parent_order].at(m)*parent_w.at(k,l,m);
                }
            }
        }
        #pragma omp parallel for collapse(2)
        for(int s=0;s<n_samples;s++){
            for(int k=0;k<parent_w.nx();k++){
                for(int l=0;l<parent_w.ny();l++){
                    for(int m=0;m<parent_w.nz();m++){
                        b_subsystem_vec_sample.at(k,s)+=r_env_sample[parent_order].at(l,s)*u_env_sample[parent_order].at(m,s)*parent_w.at(k,l,m);
                    }
                }
            }
        }
    }
    else{
        #pragma omp parallel for
        for(int l=0;l<parent_w.ny();l++){
            for(int k=0;k<parent_w.nx();k++){
                for(int m=0;m<parent_w.nz();m++){
                    b_subsystem_vec_z.at(l)+=l_env_z[parent_order].at(k)*u_env_z[parent_order].at(m)*parent_w.at(k,l,m);
                }
            }
        }
        #pragma omp parallel for collapse(2)
        for(int s=0;s<n_samples;s++){
            for(int l=0;l<parent_w.ny();l++){
                for(int k=0;k<parent_w.nx();k++){
                    for(int m=0;m<parent_w.nz();m++){
                        b_subsystem_vec_sample.at(l,s)+=l_env_sample[parent_order].at(k,s)*u_env_sample[parent_order].at(m,s)*parent_w.at(k,l,m);
                    }
                }
            }
        }
    }
    double z=0;
    for(int k=0;k<current_w.nz();k++){
        z+=a_subsystem_vec_z.at(k)*b_subsystem_vec_z.at(k);
    }
    z=log(z);
    double s_a=0;
    double s_b=0;
    double s_ab=0;
    #pragma omp parallel for reduction(-:s_a,s_b,s_ab)
    for(int s=0;s<n_samples;s++){
        double a_subsystem_sample=0;
        double b_subsystem_sample=0;
        double ab_subsystem_sample=0;
        for(int k=0;k<current_w.nz();k++){
            a_subsystem_sample+=a_subsystem_vec_sample.at(k,s)*b_subsystem_vec_z.at(k);
            b_subsystem_sample+=a_subsystem_vec_z.at(k)*b_subsystem_vec_sample.at(k,s);
            ab_subsystem_sample+=a_subsystem_vec_sample.at(k,s)*b_subsystem_vec_sample.at(k,s);
        }
        if(a_subsystem_sample<=0){a_subsystem_sample=1e-300;}
        if(b_subsystem_sample<=0){b_subsystem_sample=1e-300;}
        if(ab_subsystem_sample<=0){ab_subsystem_sample=1e-300;}
        s_a-=log(a_subsystem_sample);
        s_b-=log(b_subsystem_sample);
        s_ab-=log(ab_subsystem_sample);
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

int inner_nmf(array3d<double>& fused_mat,array3d<double>& mat1,array3d<double>& mat2,int r_max,bool compress_r){
    int r;
    if(compress_r){
        r=1;
        int upper_bound_r_max=(fused_mat.nx()<fused_mat.ny())?fused_mat.nx():fused_mat.ny();
        while(r<=((upper_bound_r_max<r_max)?upper_bound_r_max:r_max)){
            mat1=array3d<double>(fused_mat.nx(),r,1);
            mat2=array3d<double>(r,fused_mat.ny(),1);
            double kl=nmf(fused_mat,mat1,mat2,r); //nmf factors stored in mat1,mat2
            // std::cout<<r<<" "<<kl<<"\n";
            if(kl<1e-12){break;}
            if(r==((upper_bound_r_max<r_max)?upper_bound_r_max:r_max)){break;}
            r++;
        }
    }
    else{
        r=(fused_mat.nx()<fused_mat.ny())?fused_mat.nx():fused_mat.ny(); //max rank is min(row rank, col rank)
        r=(r<r_max)?r:r_max;
        mat1=array3d<double>(fused_mat.nx(),r,1);
        mat2=array3d<double>(r,fused_mat.ny(),1);
        double kl=nmf(fused_mat,mat1,mat2,r); //nmf factors stored in mat1,mat2
    }
    return r;
}

template<typename cmp>
void inner_updates(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels,int single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    z=calc_z(g,l_env_z,r_env_z,u_env_z); //also calculate envs
    w=calc_w(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
    
    // normalize_using_z(current.w(),pow(z,0.5));
    // normalize_using_z(parent.w(),pow(z,0.5));
    // z=update_cache_z(current,parent,l_env_z,r_env_z,u_env_z);
    
    array3d<double> current_m(current.w().nx(),current.w().ny(),current.w().nz());
    array3d<double> current_v(current.w().nx(),current.w().ny(),current.w().nz());
    array3d<double> parent_m(parent.w().nx(),parent.w().ny(),parent.w().nz());
    array3d<double> parent_v(parent.w().nx(),parent.w().ny(),parent.w().nz());
    
    //single-site updates
    for(int t2=1;t2<=single_site_update_count;t2++){
        optimize::site_update(current,z,w,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample,current_m,current_v,t2,lr,beta1,beta2,epsilon);

        z=update_cache_z(current,parent,l_env_z,r_env_z,u_env_z);
        w=update_cache_w(current,parent,l_env_sample,r_env_sample,u_env_sample);
        
        // normalize_using_z(current.w(),z);
        
        g.vs()[current.order()].p_bond()=current;
        g.es().erase(it);
        it=g.es().insert(current);
        
        z=update_cache_z(current,parent,l_env_z,r_env_z,u_env_z);
        w=update_cache_w(current,parent,l_env_sample,r_env_sample,u_env_sample);
        
        optimize::site_update(parent,z,w,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample,parent_m,parent_v,t2,lr,beta1,beta2,epsilon);
        
        z=update_cache_z(current,parent,l_env_z,r_env_z,u_env_z);
        w=update_cache_w(current,parent,l_env_sample,r_env_sample,u_env_sample);
        
        // normalize_using_z(parent.w(),z);
        
        g.vs()[parent.order()].p_bond()=parent;
        g.es().erase(it_parent);
        it_parent=g.es().insert(parent);
        
        z=update_cache_z(current,parent,l_env_z,r_env_z,u_env_z);
        w=update_cache_w(current,parent,l_env_sample,r_env_sample,u_env_sample);
    }
}
template void inner_updates(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,double,double,double,double);

double inner_bmi(bond& current,bond& parent,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample){
    //calculate bmi using improved tensors
    // double bmi1=fabs(optimize::calc_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample)); //take abs
    double bmi=optimize::calc_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample); //take abs
    // if(bmi<-log(current.w().nz())){bmi=2*log(current.w().nz());}
    // if(bmi<-1e-8){bmi=2*log(current.w().nz());}
    // if(bmi<-1e-4){bmi=std::numeric_limits<double>::quiet_NaN();}
    if(bmi<0){bmi=0;}
    return bmi;
}

template<typename cmp>
double way1(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample,array4d<double>& fused,int r_max,bool compress_r,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels,int single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    //split into two 3-leg tensors again via NMF: way 1
    array3d<double> fused_mat(fused.nx()*fused.ny(),fused.nz()*fused.nw(),1);
    #pragma omp parallel for collapse(4)
    for(int i=0;i<fused.nx();i++){
        for(int j=0;j<fused.ny();j++){
            for(int k=0;k<fused.nz();k++){
                for(int l=0;l<fused.nw();l++){
                    fused_mat.at((fused.ny()*i)+j,(fused.nw()*k)+l,0)=fused.at(i,j,k,l); //(ij)(kl) pairing
                }
            }
        }
    }
    
    array3d<double> mat1;
    array3d<double> mat2;
    int r=inner_nmf(fused_mat,mat1,mat2,r_max,compress_r);
    
    int current_order=current.order();
    int parent_v1=parent.v1();
    array3d<double> current_w=array3d<double>(fused.nx(),fused.ny(),r);
    array3d<double> parent_w=(parent.v1()==current.order())?array3d<double>(r,fused.nz(),fused.nw()):array3d<double>(fused.nz(),r,fused.nw());
    #pragma omp parallel for collapse(2)
    for(int i=0;i<mat1.nx();i++){
        for(int j=0;j<mat1.ny();j++){
            double val=mat1.at(i,j,0);
            current_w.at(i/current_w.ny(),i%current_w.ny(),j)=(val>epsilon)?val:epsilon;
        }
    }
    #pragma omp parallel for collapse(2)
    for(int i=0;i<mat2.nx();i++){
        for(int j=0;j<mat2.ny();j++){
            double val;
            if(parent_v1==current_order){
                val=mat2.at(i,j,0);
                parent_w.at(i,j/parent_w.nz(),j%parent_w.nz())=(val>epsilon)?val:epsilon;
            }
            else{
                val=mat2.at(i,j,0);
                parent_w.at(j/parent_w.nz(),i,j%parent_w.nz())=(val>epsilon)?val:epsilon;
            }
        }
    }
    current.w()=current_w;
    parent.w()=parent_w;
    
    g.vs()[current.order()].rank()=r;
    
    g.es().erase(it);
    it=g.es().insert(current);
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    
    inner_updates(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    double bmi=inner_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way1: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    current.ee()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    g.vs()[current.order()].ee()=bmi;
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    return bmi;
}
template double way1(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,array4d<double>&,int,bool,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,double,double,double,double);

template<typename cmp>
double way2(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample,array4d<double>& fused,int r_max,bool compress_r,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels,int single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    //split into two 3-leg tensors again via NMF: way 2
    array3d<double> fused_mat(fused.nx()*fused.nz(),fused.ny()*fused.nw(),1);
    #pragma omp parallel for collapse(4)
    for(int i=0;i<fused.nx();i++){
        for(int j=0;j<fused.nz();j++){
            for(int k=0;k<fused.ny();k++){
                for(int l=0;l<fused.nw();l++){
                    fused_mat.at((fused.nz()*i)+j,(fused.nw()*k)+l,0)=fused.at(i,k,j,l); //(ik)(jl) pairing
                }
            }
        }
    }
    
    array3d<double> mat1;
    array3d<double> mat2;
    int r=inner_nmf(fused_mat,mat1,mat2,r_max,compress_r);
    
    //fix neighbors
    int swap;
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
    g.vs()[current.order()].l_idx()=current.v1();
    g.vs()[current.order()].r_idx()=current.v2();
    g.vs()[parent.v1()].u_idx()=parent.order();
    g.vs()[parent.v2()].u_idx()=parent.order();
    g.vs()[parent.order()].l_idx()=parent.v1();
    g.vs()[parent.order()].r_idx()=parent.v2();
    
    int current_order=current.order();
    int parent_v1=parent.v1();
    array3d<double> current_w=array3d<double>(fused.nx(),fused.nz(),r);
    array3d<double> parent_w=(parent.v1()==current.order())?array3d<double>(r,fused.ny(),fused.nw()):array3d<double>(fused.ny(),r,fused.nw());
    g.vs()[current.order()].rank()=r;
    #pragma omp parallel for collapse(2)
    for(int i=0;i<mat1.nx();i++){
        for(int j=0;j<mat1.ny();j++){
            double val=mat1.at(i,j,0);
            current_w.at(i/current_w.ny(),i%current_w.ny(),j)=(val>epsilon)?val:epsilon;
        }
    }
    #pragma omp parallel for collapse(2)
    for(int i=0;i<mat2.nx();i++){
        for(int j=0;j<mat2.ny();j++){
            double val;
            val=mat2.at(i,j,0);
            if(parent_v1==current_order){
                parent_w.at(i,j/parent_w.nz(),j%parent_w.nz())=(val>epsilon)?val:epsilon;
            }
            else{
                parent_w.at(j/parent_w.nz(),i,j%parent_w.nz())=(val>epsilon)?val:epsilon;
            }
        }
    }
    current.w()=current_w;
    parent.w()=parent_w;
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
        d_it=g.es().lower_bound(key);
    }
    // std::cout<<(std::string) g<<"\n";
    
    inner_updates(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    double bmi=inner_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way2: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    current.ee()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    g.vs()[current.order()].ee()=bmi;
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    return bmi;
}
template double way2(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,array4d<double>&,int,bool,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,double,double,double,double);

template<typename cmp>
double way3(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<array1d<double> >& l_env_z,std::vector<array1d<double> >& r_env_z,std::vector<array1d<double> >& u_env_z,std::vector<double>& w,std::vector<array2d<double> >& l_env_sample,std::vector<array2d<double> >& r_env_sample,std::vector<array2d<double> >& u_env_sample,array4d<double>& fused,int r_max,bool compress_r,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels,int single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    //split into two 3-leg tensors again via NMF: way 2
    array3d<double> fused_mat(fused.nz()*fused.ny(),fused.nx()*fused.nw(),1);
    #pragma omp parallel for collapse(4)
    for(int i=0;i<fused.nz();i++){
        for(int j=0;j<fused.ny();j++){
            for(int k=0;k<fused.nx();k++){
                for(int l=0;l<fused.nw();l++){
                    fused_mat.at((fused.ny()*i)+j,(fused.nw()*k)+l,0)=fused.at(k,j,i,l); //(kj)(il) pairing
                }
            }
        }
    }
    
    array3d<double> mat1;
    array3d<double> mat2;
    int r=inner_nmf(fused_mat,mat1,mat2,r_max,compress_r);
    
    //fix neighbors
    int swap;
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
    g.vs()[current.order()].l_idx()=current.v1();
    g.vs()[current.order()].r_idx()=current.v2();
    g.vs()[parent.v1()].u_idx()=parent.order();
    g.vs()[parent.v2()].u_idx()=parent.order();
    g.vs()[parent.order()].l_idx()=parent.v1();
    g.vs()[parent.order()].r_idx()=parent.v2();
    
    int current_order=current.order();
    int parent_v1=parent.v1();
    array3d<double> current_w=array3d<double>(fused.nz(),fused.ny(),r);
    array3d<double> parent_w=(parent.v1()==current.order())?array3d<double>(r,fused.nx(),fused.nw()):array3d<double>(fused.nx(),r,fused.nw());
    g.vs()[current.order()].rank()=r;
    #pragma omp parallel for collapse(2)
    for(int i=0;i<mat1.nx();i++){
        for(int j=0;j<mat1.ny();j++){
            double val=mat1.at(i,j,0);
            current_w.at(i/current_w.ny(),i%current_w.ny(),j)=(val>epsilon)?val:epsilon;
        }
    }
    #pragma omp parallel for collapse(2)
    for(int i=0;i<mat2.nx();i++){
        for(int j=0;j<mat2.ny();j++){
            double val;
            val=mat2.at(i,j,0);
            if(parent_v1==current_order){
                parent_w.at(i,j/parent_w.nz(),j%parent_w.nz())=(val>epsilon)?val:epsilon;
            }
            else{
                parent_w.at(j/parent_w.nz(),i,j%parent_w.nz())=(val>epsilon)?val:epsilon;
            }
        }
    }
    current.w()=current_w;
    parent.w()=parent_w;
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
        d_it=g.es().lower_bound(key);
    }
    // std::cout<<(std::string) g<<"\n";
    
    inner_updates(g,it,it_parent,current,parent,z,l_env_z,r_env_z,u_env_z,w,l_env_sample,r_env_sample,u_env_sample,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    double bmi=inner_bmi(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way3: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    current.ee()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    g.vs()[current.order()].ee()=bmi;
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    return bmi;
}
template double way3(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<double>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,array4d<double>&,int,bool,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,int,double,double,double,double);