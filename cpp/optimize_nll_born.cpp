#include <limits>
#include <list>
#include <random>

#include "omp.h"

#include "algorithm_nll.hpp"
#include "mat_ops.hpp"
#include "mpi_utils.hpp"
#include "optimize_nll_born.hpp"
#include "ttn_ops.hpp"
#include "ttn_ops_born.hpp"

void aux_update_lr_cache_born(bond& current,bond& parent,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample){
    #pragma omp parallel for
    for(size_t s=0;s<l_env_sample[current.order()].size();s++){
        array1d<double> res_vec_sample(current.w().nz());
        for(size_t k=0;k<current.w().nz();k++){
            std::vector<double> res_vec_sample_addends;
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    res_vec_sample_addends.push_back(l_env_sample[current.order()][s].at(i)*r_env_sample[current.order()][s].at(j)*current.w().at(i,j,k));
                }
            }
            res_vec_sample.at(k)=vec_add_float(res_vec_sample_addends);
        }
        if(parent.v1()==current.order()){
            l_env_sample[parent.order()][s]=res_vec_sample;
        }
        else{
            r_env_sample[parent.order()][s]=res_vec_sample;
        }
    }
}

void aux_update_u_cache_born(bond& current,bond& parent,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample){
    #pragma omp parallel for
    for(size_t s=0;s<u_env_sample[parent.order()].size();s++){
        array1d<double> res_vec2_sample(current.w().nz());
        if(parent.v1()==current.order()){
            for(size_t i=0;i<parent.w().nx();i++){
                std::vector<double> res_vec2_sample_addends;
                for(size_t j=0;j<parent.w().ny();j++){
                    for(size_t k=0;k<parent.w().nz();k++){
                        res_vec2_sample_addends.push_back(r_env_sample[parent.order()][s].at(j)*u_env_sample[parent.order()][s].at(k)*parent.w().at(i,j,k));
                    }
                }
                res_vec2_sample.at(i)=vec_add_float(res_vec2_sample_addends);
            }
        }
        else{
            for(size_t j=0;j<parent.w().ny();j++){
                std::vector<double> res_vec2_sample_addends;
                for(size_t i=0;i<parent.w().nx();i++){
                    for(size_t k=0;k<parent.w().nz();k++){
                        res_vec2_sample_addends.push_back(l_env_sample[parent.order()][s].at(i)*u_env_sample[parent.order()][s].at(k)*parent.w().at(i,j,k));
                    }
                }
                res_vec2_sample.at(j)=vec_add_float(res_vec2_sample_addends);
            }
        }
        u_env_sample[current.order()][s]=res_vec2_sample;
    }
}

template<typename cmp>
double inner_bmi_born(graph<cmp>& g,bond& current,bond& parent,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample){
    //calculate bmi using improved tensors
    // double bmi1=fabs(optimize::calc_bmi_born(g,current,w,l_env_sample,r_env_sample,u_env_sample)); //take abs
    double bmi=optimize::calc_bmi_born(g,current,w,l_env_sample,r_env_sample,u_env_sample); //take abs
    // double bmi=optimize::calc_ee_born(current,parent); //take abs
    // if(bmi<-log(current.w().nz())){bmi=2*log(current.w().nz());}
    // if(bmi<-1e-8){bmi=2*log(current.w().nz());}
    if(bmi<-1e-4){bmi=std::numeric_limits<double>::quiet_NaN();}
    return bmi;
}
template double inner_bmi_born(graph<bmi_comparator>&,bond&,bond&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

template<typename cmp>
double optimize::opt_struct_nll_born(graph<cmp>& g,std::vector<sample_data>& train_samples,std::vector<size_t>& train_labels,std::vector<sample_data>& test_samples,std::vector<size_t>& test_labels,size_t iter_max,size_t r_max,bool compress_r,double lr,size_t batch_size,std::map<size_t,double>& train_nll_history,std::map<size_t,double>& test_nll_history,std::map<size_t,size_t>& sweep_history,bool struct_opt){
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
    //initialize minibatches
    size_t batch_start_idx=0;
    std::vector<size_t> batch_shuffle(train_samples.size());
    for(size_t i=0;i<train_samples.size();i++){
        batch_shuffle[i]=i;
    }
    std::vector<sample_data> train_samples_batch(train_samples.begin()+batch_start_idx,((batch_start_idx+batch_size<train_samples.size())?(train_samples.begin()+batch_start_idx+batch_size):train_samples.end()));
    std::vector<size_t> train_labels_batch;
    if(train_labels.size()!=0){
        std::vector<size_t> train_labels_batch(train_labels.begin()+batch_start_idx,((batch_start_idx+batch_size<train_labels.size())?(train_labels.begin()+batch_start_idx+batch_size):train_labels.end()));
    }
    batch_start_idx=(batch_start_idx+batch_size<train_samples.size())?batch_start_idx+batch_size:0;
    
    std::vector<std::vector<array1d<double> > > l_env_sample;
    std::vector<std::vector<array1d<double> > > r_env_sample;
    std::vector<std::vector<array1d<double> > > u_env_sample;
    //canonicalize ttn
    canonicalize(g,(*g.es().begin()).order());
    double z=calc_z_born(g);
    std::vector<double> w=calc_w_born(g,train_samples_batch,train_labels_batch,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
    
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
        canonicalize(g,(*g.es().begin()).order());
        z=calc_z_born(g);
        w=calc_w_born(g,train_samples_batch,train_labels_batch,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
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
            // std::cout<<"g:"<<(std::string) g<<"\n";
            // std::cout<<"current:"<<(std::string) current.w()<<"\n";
            // std::cout<<"parent:"<<(std::string) parent.w()<<"\n";
            
            array4d<double> fused=optimize::fused_update_born(g,current,parent,z,w,l_env_sample,r_env_sample,u_env_sample,fused_m,fused_v,t,lr,beta1,beta2,epsilon);
            
            //save state of graph, iterators, and caches before proposing changes
            graph<cmp> orig_g=g;
            bond orig_current=current;
            bond orig_parent=parent;
            double orig_z=z;
            std::vector<double> orig_w=w;
            std::vector<std::vector<array1d<double> > > orig_l_env_sample=l_env_sample;
            std::vector<std::vector<array1d<double> > > orig_r_env_sample=r_env_sample;
            std::vector<std::vector<array1d<double> > > orig_u_env_sample=u_env_sample;
            
            double bmi1=way1_born(g,it,it_parent,current,parent,z,w,l_env_sample,r_env_sample,u_env_sample,fused,r_max,compress_r,train_samples_batch,train_labels_batch,single_site_update_count,lr,beta1,beta2,epsilon);
            
            if(struct_opt){
                graph<cmp> way1_g=g;
                bond way1_current=current;
                bond way1_parent=parent;
                double way1_z=z;
                std::vector<double> way1_w=w;
                std::vector<std::vector<array1d<double> > > way1_l_env_sample=l_env_sample;
                std::vector<std::vector<array1d<double> > > way1_r_env_sample=r_env_sample;
                std::vector<std::vector<array1d<double> > > way1_u_env_sample=u_env_sample;
                
                g=orig_g;
                current=orig_current;
                parent=orig_parent;
                z=orig_z;
                w=orig_w;
                l_env_sample=orig_l_env_sample;
                r_env_sample=orig_r_env_sample;
                u_env_sample=orig_u_env_sample;
                it=g.es().find(current);
                it_parent=g.es().find(parent);
                
                double bmi2=way2_born(g,it,it_parent,current,parent,z,w,l_env_sample,r_env_sample,u_env_sample,fused,r_max,compress_r,train_samples_batch,train_labels_batch,single_site_update_count,lr,beta1,beta2,epsilon);
                
                graph<bmi_comparator> way2_g=g;
                bond way2_current=current;
                bond way2_parent=parent;
                double way2_z=z;
                std::vector<double> way2_w=w;
                std::vector<std::vector<array1d<double> > > way2_l_env_sample=l_env_sample;
                std::vector<std::vector<array1d<double> > > way2_r_env_sample=r_env_sample;
                std::vector<std::vector<array1d<double> > > way2_u_env_sample=u_env_sample;
                
                g=orig_g;
                current=orig_current;
                parent=orig_parent;
                z=orig_z;
                w=orig_w;
                l_env_sample=orig_l_env_sample;
                r_env_sample=orig_r_env_sample;
                u_env_sample=orig_u_env_sample;
                it=g.es().find(current);
                it_parent=g.es().find(parent);
                
                double bmi3=way3_born(g,it,it_parent,current,parent,z,w,l_env_sample,r_env_sample,u_env_sample,fused,r_max,compress_r,train_samples_batch,train_labels_batch,single_site_update_count,lr,beta1,beta2,epsilon);
                
                graph<bmi_comparator> way3_g=g;
                bond way3_current=current;
                bond way3_parent=parent;
                double way3_z=z;
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
                    w=way3_w;
                    l_env_sample=way3_l_env_sample;
                    r_env_sample=way3_r_env_sample;
                    u_env_sample=way3_u_env_sample;
                    it=g.es().find(current);
                    it_parent=g.es().find(parent);
                    // std::cout<<"selected bmi3\n";
                }
            }
            
            //calculate entanglement entropy
            current.ee()=optimize::calc_ee_born(current,parent);
            // current.ee()=inner_bmi_born(g,current,parent,w,l_env_sample,r_env_sample,u_env_sample);
            
            done_idxs.insert(current.order());
            // std::cout<<(std::string)g<<"\n";
            
            g.vs()[current.order()].p_bond()=current;
            g.vs()[parent.order()].p_bond()=parent;
            
            g.vs()[current.order()].rank()=current.w().nz();
            g.vs()[parent.order()].rank()=parent.w().nz();
            
            g.es().erase(it);
            it=g.es().insert(current);
            g.es().erase(it_parent);
            it_parent=g.es().insert(parent);
            
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
            auto it_next=g.es().lower_bound(key);
            bond next_current=*it_next;
            //qr decompose the current weight tensor with not-next legs fused, then push r matrix towards next tensor
            if((g.vs()[current.v1()].depth()!=0)&&(done_idxs.find(current.v1())==done_idxs.end())){
                // std::cout<<iter<<" L\n";
                array3d<double> target=matricize(current.w(),0);
                //compute qr
                array3d<double> q;
                array3d<double> r;
                size_t status=0;
                qr(target,q,r,status);
                current.w()=tensorize(q,current.w().ny(),current.w().nz(),0);
                array3d<double> res(next_current.w().nx(),next_current.w().ny(),r.nx());
                for(size_t i=0;i<next_current.w().nx();i++){
                    for(size_t j=0;j<next_current.w().ny();j++){
                        for(size_t k=0;k<r.nx();k++){
                            std::vector<double> res_addends;
                            for(size_t l=0;l<r.ny();l++){
                                res_addends.push_back(next_current.w().at(i,j,l)*r.at(k,l,0));
                            }
                            res.at(i,j,k)=vec_add_float(res_addends);
                        }
                    }
                }
                next_current.w()=res;
            }
            else if((g.vs()[current.v2()].depth()!=0)&&(done_idxs.find(current.v2())==done_idxs.end())){
                // std::cout<<iter<<" R\n";
                array3d<double> target=matricize(current.w(),1);
                //compute qr
                array3d<double> q;
                array3d<double> r;
                size_t status=0;
                qr(target,q,r,status);
                current.w()=tensorize(q,current.w().nx(),current.w().nz(),1);
                array3d<double> res(next_current.w().nx(),next_current.w().ny(),r.nx());
                for(size_t i=0;i<next_current.w().nx();i++){
                    for(size_t j=0;j<next_current.w().ny();j++){
                        for(size_t k=0;k<r.nx();k++){
                            std::vector<double> res_addends;
                            for(size_t l=0;l<r.ny();l++){
                                res_addends.push_back(next_current.w().at(i,j,l)*r.at(k,l,0));
                            }
                            res.at(i,j,k)=vec_add_float(res_addends);
                        }
                    }
                }
                next_current.w()=res;
            }
            else{
                // std::cout<<iter<<" U\n";
                array3d<double> target=matricize(current.w(),2);
                //compute qr
                array3d<double> q;
                array3d<double> r;
                size_t status=0;
                qr(target,q,r,status);
                current.w()=tensorize(q,current.w().nx(),current.w().ny(),2);
                array3d<double> res;
                if(current.order()==next_current.v1()){
                    res=array3d<double>(r.nx(),next_current.w().ny(),next_current.w().nz());
                    for(size_t i=0;i<r.nx();i++){
                        for(size_t j=0;j<next_current.w().ny();j++){
                            for(size_t k=0;k<next_current.w().nz();k++){
                                std::vector<double> res_addends;
                                for(size_t l=0;l<r.ny();l++){
                                    res_addends.push_back(next_current.w().at(l,j,k)*r.at(i,l,0));
                                }
                                res.at(i,j,k)=vec_add_float(res_addends);
                            }
                        }
                    }
                }
                else{
                    res=array3d<double>(next_current.w().nx(),r.nx(),next_current.w().nz());
                    for(size_t i=0;i<next_current.w().nx();i++){
                        for(size_t j=0;j<r.nx();j++){
                            for(size_t k=0;k<next_current.w().nz();k++){
                                std::vector<double> res_addends;
                                for(size_t l=0;l<r.ny();l++){
                                    res_addends.push_back(next_current.w().at(i,l,k)*r.at(j,l,0));
                                }
                                res.at(i,j,k)=vec_add_float(res_addends);
                            }
                        }
                    }
                }
                next_current.w()=res;
            }
            
            normalize(next_current.w()); //only normalize r
            
            g.vs()[current.order()].p_bond()=current;
            g.es().erase(it);
            it=g.es().insert(current);
            g.vs()[next_current.order()].p_bond()=next_current;
            g.es().erase(it_next);
            it_next=g.es().insert(next_current);
            
            g.vs()[current.order()].rank()=current.w().nz();
            g.vs()[next_current.order()].rank()=next_current.w().nz();
            g.center_idx()=next_current.order();
            it=it_next;
            
            z=calc_z_born(g);
            w=calc_w_born(g,train_samples_batch,train_labels_batch,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
            
            // std::cout<<"new g:"<<(std::string) g<<"\n";
            // std::cout<<"new current:"<<(std::string) current.w()<<"\n";
            // std::cout<<"new parent:"<<(std::string) parent.w()<<"\n";
            
            if(iter==iter_max){break;}
            iter++;
        }
        //calculate train nll
        nll=0;
        #pragma omp parallel for reduction(-:nll)
        for(size_t s=0;s<train_samples.size();s++){
            nll-=log(w[s]);
        }
        nll/=(double) train_samples.size();
        nll+=log(z);
        train_nll_history.insert(std::pair<size_t,double>(iter,nll));
            
        //calculate test nll per sweep
        if(test_samples.size()!=0){
            std::vector<std::vector<array1d<double> > > test_l_env_sample;
            std::vector<std::vector<array1d<double> > > test_r_env_sample;
            std::vector<std::vector<array1d<double> > > test_u_env_sample;
            std::vector<double> test_w=calc_w_born(g,test_samples,test_labels,test_l_env_sample,test_r_env_sample,test_u_env_sample);
            test_nll=0;
            #pragma omp parallel for reduction(-:test_nll)
            for(size_t s=0;s<test_samples.size();s++){
                test_nll-=log(test_w[s]);
            }
            test_nll/=(double) test_samples.size();
            test_nll+=log(z);
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
            if(test_samples.size()!=0){
                std::cout<<"sweep "<<t<<" iter "<<iter<<" train nll="<<nll<<" test nll="<<test_nll<<"\n";
            }
            else{
                std::cout<<"sweep "<<t<<" iter "<<iter<<" train nll="<<nll<<"\n";
            }
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
        
        //update minibatch and recalculate minibatch-dependent w
        train_samples_batch=std::vector<sample_data>(train_samples.begin()+batch_start_idx,((batch_start_idx+batch_size<train_samples.size())?(train_samples.begin()+batch_start_idx+batch_size):train_samples.end()));
        if(train_labels.size()!=0){
            train_labels_batch=std::vector<size_t>(train_labels.begin()+batch_start_idx,((batch_start_idx+batch_size<train_labels.size())?(train_labels.begin()+batch_start_idx+batch_size):train_labels.end()));
        }
        batch_start_idx=(batch_start_idx+batch_size<train_samples.size())?batch_start_idx+batch_size:0;
        if(batch_start_idx==0){
            std::shuffle(std::begin(batch_shuffle),std::end(batch_shuffle),mpi_utils::prng);
            std::vector<sample_data> train_samples_copy=train_samples;
            for(size_t i=0;i<train_samples.size();i++){
                train_samples[i]=train_samples_copy[batch_shuffle[i]];
            }
            if(train_labels.size()!=0){
                std::vector<size_t> train_labels_copy=train_labels;
                for(size_t i=0;i<train_labels.size();i++){
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
    if(test_samples.size()!=0){std::cout<<"best test nll="<<best_test_nll<<"\n";}
    t++;
    nll=best_nll;
    test_nll=best_test_nll;
    g.vs()=best_vs;
    g.es()=best_es;
    
    z=calc_z_born(g);
    w=calc_w_born(g,train_samples,train_labels,l_env_sample,r_env_sample,u_env_sample);
    
    return best_nll;
}
template double optimize::opt_struct_nll_born(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<sample_data>&,std::vector<size_t>&,size_t,size_t,bool,double,size_t,std::map<size_t,double>&,std::map<size_t,double>&,std::map<size_t,size_t>&,bool);

template<typename cmp>
std::vector<size_t> optimize::classify_born(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<array1d<double> >& probs){
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
                        if(a==(samples[s].s()[n]-1)){ //if a==samples[s].s()[n]-1, element is 1. else 0
                            vec_e.at(a)=1;
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
                                res_vec_addends.push_back(contracted_vectors[(*it).v1()][s].at(i)*contracted_vectors[(*it).v2()][s].at(j)*(*it).w().at(i,j,k));
                            }
                        }
                        res_vec[s].at(k)=vec_add_float(res_vec_addends);
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
        for(size_t i=0;i<probs[s].nx();i++){
            probs[s].at(i)*=probs[s].at(i);
        }
    }
    #pragma omp parallel for
    for(size_t s=0;s<n_samples;s++){
        double prob_sum=vec_add_float(probs[s].e());
        for(size_t i=0;i<probs[s].nx();i++){
            probs[s].at(i)/=prob_sum;
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
template std::vector<size_t> optimize::classify_born(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<array1d<double> >&);

template<typename cmp>
void optimize::site_update_born(graph<cmp>& g,bond& b,double z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,array3d<double>& m_cache,array3d<double>& v_cache,size_t t,double lr,double beta1,double beta2,double epsilon){
    array3d<double> dz=g.vs()[g.center_idx()].p_bond().w();
    std::vector<array3d<double> > dw(w.size());
    #pragma omp parallel for
    for(size_t s=0;s<w.size();s++){
        array3d<double> dw_e(b.w().nx(),b.w().ny(),b.w().nz());
        for(size_t i=0;i<dw_e.nx();i++){
            for(size_t j=0;j<dw_e.ny();j++){
                for(size_t k=0;k<dw_e.nz();k++){
                    dw_e.at(i,j,k)=l_env_sample[b.order()][s].at(i)*r_env_sample[b.order()][s].at(j)*u_env_sample[b.order()][s].at(k);
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
                grad_z_term.at(i,j,k)=dz.at(i,j,k)/z;
                std::vector<double> grad_w_term_addends(w.size());
                #pragma omp parallel for
                for(size_t s=0;s<w.size();s++){
                    grad_w_term_addends[s]=dw[s].at(i,j,k)/sqrt(w[s]);
                }
                grad_w_term.at(i,j,k)=vec_add_float(grad_w_term_addends)/(double) w.size();
                //perform unconstrained gd on original problem
                grad.at(i,j,k)=(2*grad_z_term.at(i,j,k))-(2*grad_w_term.at(i,j,k));
            }
        }
    }
    std::vector<double> grad_norm_addends;
    for(size_t i=0;i<grad.nx();i++){
        for(size_t j=0;j<grad.ny();j++){
            for(size_t k=0;k<grad.nz();k++){
                grad_norm_addends.push_back(grad.at(i,j,k)*grad.at(i,j,k));
            }
        }
    }
    double grad_norm=sqrt(vec_add_float(grad_norm_addends));
    if(grad_norm>1){ //gradient clipping by norm
        for(size_t i=0;i<grad.nx();i++){
            for(size_t j=0;j<grad.ny();j++){
                for(size_t k=0;k<grad.nz();k++){
                    grad.at(i,j,k)=1*(grad.at(i,j,k)/grad_norm);
                }
            }
        }
    }
    for(size_t i=0;i<grad.nx();i++){
        for(size_t j=0;j<grad.ny();j++){
            for(size_t k=0;k<grad.nz();k++){
                m_cache.at(i,j,k)=(beta1*m_cache.at(i,j,k))+((1-beta1)*grad.at(i,j,k));
                v_cache.at(i,j,k)=(beta2*v_cache.at(i,j,k))+((1-beta2)*grad.at(i,j,k)*grad.at(i,j,k));
                double corrected_m=m_cache.at(i,j,k)/(1-pow(beta1,(double) t));
                double corrected_v=v_cache.at(i,j,k)/(1-pow(beta2,(double) t));
                b.w().at(i,j,k)=b.w().at(i,j,k)-(lr*0.01*b.w().at(i,j,k)); //weight decay (adamw)
                b.w().at(i,j,k)=b.w().at(i,j,k)-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                // b.w().at(i,j,k)=b.w().at(i,j,k)-(lr*grad.at(i,j,k));
            }
        }
    }
    normalize(b.w());
}
template void optimize::site_update_born(graph<bmi_comparator>&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array3d<double>&,array3d<double>&,size_t,double,double,double,double);

template<typename cmp>
array4d<double> optimize::fused_update_born(graph<cmp>& g,bond& b1,bond& b2,double z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,std::map<std::pair<size_t,size_t>,array4d<double> >& m_cache,std::map<std::pair<size_t,size_t>,array4d<double> >& v_cache,size_t t,double lr,double beta1,double beta2,double epsilon){
    //calculate fused, dz and dw
    array4d<double> fused(b1.w().nx(),b1.w().ny(),(b2.v1()==b1.order())?b2.w().ny():b2.w().nx(),b2.w().nz());
    for(size_t i=0;i<fused.nx();i++){
        for(size_t j=0;j<fused.ny();j++){
            for(size_t k=0;k<fused.nz();k++){
                for(size_t l=0;l<fused.nw();l++){
                    std::vector<double> sum_addends;
                    for(size_t m=0;m<b1.w().nz();m++){
                        sum_addends.push_back(b1.w().at(i,j,m)*((b2.v1()==b1.order())?b2.w().at(m,k,l):b2.w().at(k,m,l)));
                    }
                    fused.at(i,j,k,l)=vec_add_float(sum_addends);
                }
            }
        }
    }
    
    array4d<double> dz=fused;
    std::vector<array4d<double> > dw(w.size());
    #pragma omp parallel for
    for(size_t s=0;s<w.size();s++){
        array4d<double> dw_e(fused.nx(),fused.ny(),fused.nz(),fused.nw());
        for(size_t i=0;i<dw_e.nx();i++){
            for(size_t j=0;j<dw_e.ny();j++){
                for(size_t k=0;k<dw_e.nz();k++){
                    for(size_t l=0;l<dw_e.nw();l++){
                        dw_e.at(i,j,k,l)=l_env_sample[b1.order()][s].at(i)*r_env_sample[b1.order()][s].at(j)*((b2.v1()==b1.order())?r_env_sample[b2.order()][s].at(k):l_env_sample[b2.order()][s].at(k))*u_env_sample[b2.order()][s].at(l);
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
    
    // std::cout<<(std::string) b1.w()<<"\n";
    for(size_t i=0;i<grad_2site.nx();i++){
        for(size_t j=0;j<grad_2site.ny();j++){
            for(size_t k=0;k<grad_2site.nz();k++){
                for(size_t l=0;l<grad_2site.nw();l++){
                    grad_z_term_2site.at(i,j,k,l)=dz.at(i,j,k,l)/z;
                    std::vector<double> grad_w_term_2site_addends(w.size());
                    #pragma omp parallel for
                    for(size_t s=0;s<w.size();s++){
                        grad_w_term_2site_addends[s]=dw[s].at(i,j,k,l)/sqrt(w[s]);
                    }
                    grad_w_term_2site.at(i,j,k,l)=vec_add_float(grad_w_term_2site_addends)/(double) w.size();
                    //perform projected nonnegative gd on original problem
                    grad_2site.at(i,j,k,l)=(2*grad_z_term_2site.at(i,j,k,l))-(2*grad_w_term_2site.at(i,j,k,l));
                }
            }
        }
    }
    std::vector<double> grad_2site_norm_addends;
    for(size_t i=0;i<grad_2site.nx();i++){
        for(size_t j=0;j<grad_2site.ny();j++){
            for(size_t k=0;k<grad_2site.nz();k++){
                for(size_t l=0;l<grad_2site.nw();l++){
                    grad_2site_norm_addends.push_back(grad_2site.at(i,j,k,l)*grad_2site.at(i,j,k,l));
                }
            }
        }
    }
    double grad_2site_norm=sqrt(vec_add_float(grad_2site_norm_addends));
    if(grad_2site_norm>1){ //gradient clipping by norm
        for(size_t i=0;i<grad_2site.nx();i++){
            for(size_t j=0;j<grad_2site.ny();j++){
                for(size_t k=0;k<grad_2site.nz();k++){
                    for(size_t l=0;l<grad_2site.nw();l++){
                        grad_2site.at(i,j,k,l)=1*(grad_2site.at(i,j,k,l)/grad_2site_norm);
                    }
                }
            }
        }
    }
    for(size_t i=0;i<grad_2site.nx();i++){
        for(size_t j=0;j<grad_2site.ny();j++){
            for(size_t k=0;k<grad_2site.nz();k++){
                for(size_t l=0;l<grad_2site.nw();l++){
                    m_cache[key].at(i,j,k,l)=(beta1*m_cache[key].at(i,j,k,l))+((1-beta1)*grad_2site.at(i,j,k,l));
                    v_cache[key].at(i,j,k,l)=(beta2*v_cache[key].at(i,j,k,l))+((1-beta2)*grad_2site.at(i,j,k,l)*grad_2site.at(i,j,k,l));
                    double corrected_m=m_cache[key].at(i,j,k,l)/(1-pow(beta1,(double) t));
                    double corrected_v=v_cache[key].at(i,j,k,l)/(1-pow(beta2,(double) t));
                    fused.at(i,j,k,l)=fused.at(i,j,k,l)-(lr*0.01*fused.at(i,j,k,l)); //weight decay (adamw)
                    fused.at(i,j,k,l)=fused.at(i,j,k,l)-(lr*(corrected_m/(sqrt(corrected_v)+epsilon)));
                    // fused.at(i,j,k,l)=fused.at(i,j,k,l)-(lr*grad_2site.at(i,j,k,l));
                }
            }
        }
    }
    normalize(fused);
    return fused;
}
template array4d<double> optimize::fused_update_born(graph<bmi_comparator>&,bond&,bond&,double,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,std::map<std::pair<size_t,size_t>,array4d<double> >&,size_t,double,double,double,double);

template<typename cmp>
double optimize::calc_bmi_born(graph<cmp>& g,bond& current,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample){
    if(current.w().nz()==1){return 0;} //bmi is nonnegative and upper-bounded by bond dimension, so dim=1 -> bmi=0
    //assumes that the ttn is canonicalized with center at current, and p_bonds are accurate
    size_t n_samples=w.size();
    std::vector<double> a_subsystem_sample(n_samples);
    std::vector<double> b_subsystem_sample(n_samples);
    std::vector<double> ab_subsystem_sample(n_samples);
    #pragma omp parallel for
    for(size_t s=0;s<n_samples;s++){
        std::vector<double> a_subsystem_sample_addends;
        for(size_t m=0;m<current.w().nz();m++){
            std::vector<double> weight_addends;
            for(size_t k=0;k<l_env_sample[current.order()][s].nx();k++){
                for(size_t l=0;l<r_env_sample[current.order()][s].nx();l++){
                     weight_addends.push_back(l_env_sample[current.order()][s].at(k)*r_env_sample[current.order()][s].at(l)*current.w().at(k,l,m));
                }
            }
            a_subsystem_sample_addends.push_back(vec_add_float(weight_addends)*vec_add_float(weight_addends));
        }
        a_subsystem_sample[s]=vec_add_float(a_subsystem_sample_addends);
        size_t prev_idx=current.order();
        bond parent=g.vs()[g.vs()[current.order()].u_idx()].p_bond();
        std::vector<array3d<double> >b_subsystem_tensor_sample(n_samples);
        b_subsystem_tensor_sample[s]=current.w();
        while(current.order()!=g.vs().size()-1){
            array3d<double> intermediate;
            if(prev_idx==parent.v1()){
                intermediate=array3d<double>(parent.w().nx(),parent.w().nz(),1);
                for(size_t k=0;k<parent.w().nx();k++){
                    for(size_t m=0;m<parent.w().nz();m++){
                        std::vector<double> intermediate_addends;
                        for(size_t l=0;l<parent.w().ny();l++){
                            intermediate_addends.push_back(parent.w().at(k,l,m)*r_env_sample[parent.order()][s].at(l));
                        }
                        intermediate.at(k,m,0)=vec_add_float(intermediate_addends);
                    }
                }
            }
            else{
                intermediate=array3d<double>(parent.w().ny(),parent.w().nz(),1);
                for(size_t l=0;l<parent.w().ny();l++){
                    for(size_t m=0;m<parent.w().nz();m++){
                        std::vector<double> intermediate_addends;
                        for(size_t k=0;k<parent.w().nx();k++){
                            intermediate_addends.push_back(parent.w().at(k,l,m)*l_env_sample[parent.order()][s].at(k));
                        }
                        intermediate.at(l,m,0)=vec_add_float(intermediate_addends);
                    }
                }
            }
            array3d<double> res(b_subsystem_tensor_sample[s].nx(),b_subsystem_tensor_sample[s].ny(),intermediate.ny());
            for(size_t k=0;k<res.nx();k++){
                for(size_t l=0;l<res.ny();l++){
                    for(size_t m=0;m<res.nz();m++){
                        std::vector<double> res_addends;
                        for(size_t n=0;n<intermediate.nx();n++){
                            res_addends.push_back(b_subsystem_tensor_sample[s].at(k,l,n)*intermediate.at(n,m,0));
                        }
                        res.at(k,l,m)=vec_add_float(res_addends);
                    }
                }
            }
            b_subsystem_tensor_sample[s]=res;
            // std::cout<<prev_idx<<" "<<parent.order()<<"\n";
            if(parent.order()==g.vs().size()-1){break;}
            prev_idx=parent.order();
            parent=g.vs()[g.vs()[parent.order()].u_idx()].p_bond();
        }
        std::vector<double> b_subsystem_sample_addends;
        for(size_t k=0;k<b_subsystem_tensor_sample[s].nx();k++){
            for(size_t l=0;l<b_subsystem_tensor_sample[s].ny();l++){
                for(size_t m=0;m<b_subsystem_tensor_sample[s].nz();m++){
                    b_subsystem_sample_addends.push_back(b_subsystem_tensor_sample[s].at(k,l,m)*b_subsystem_tensor_sample[s].at(k,l,m));
                }
            }
        }
        b_subsystem_sample[s]=vec_add_float(b_subsystem_sample_addends);
        ab_subsystem_sample[s]=w[s];
    }
    
    std::vector<double> z_addends;
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                z_addends.push_back(current.w().at(i,j,k)*current.w().at(i,j,k));
            }
        }
    }
    double z=log(vec_add_float(z_addends));
    double s_a=0;
    double s_b=0;
    double s_ab=0;
    #pragma omp parallel for reduction(-:s_a,s_b,s_ab)
    for(size_t s=0;s<n_samples;s++){
        s_a-=log(a_subsystem_sample[s]);
        s_b-=log(b_subsystem_sample[s]);
        s_ab-=log(ab_subsystem_sample[s]);
    }
    double bmi=((s_a+s_b-s_ab)/(double) n_samples)+z;
    // std::cout<<"current:\n"<<(std::string)current.w()<<"\n";
    // std::cout<<"parent:\n"<<(std::string)parent.w()<<"\n";
    // std::cout<<"z: "<<z<<"\n";
    // std::cout<<"s_a: "<<s_a<<"\n";
    // std::cout<<"s_b: "<<s_b<<"\n";
    // std::cout<<"s_ab: "<<s_ab<<"\n";
    // std::cout<<"("<<current.order()<<","<<g.vs()[current.order()].u_idx()<<") bmi: "<<bmi<<"\n";
    return bmi;
}
template double optimize::calc_bmi_born(graph<bmi_comparator>&,bond&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

double optimize::calc_ee_born(bond& current,bond& parent){
    if(current.w().nz()==1){return 0;} //ee is nonnegative and upper-bounded by bond dimension, so dim=1 -> ee=0
    double ee=0;
    array4d<double> fused(current.w().nx(),current.w().ny(),(parent.v1()==current.order())?parent.w().ny():parent.w().nx(),parent.w().nz());
    for(size_t i=0;i<fused.nx();i++){
        for(size_t j=0;j<fused.ny();j++){
            for(size_t k=0;k<fused.nz();k++){
                for(size_t l=0;l<fused.nw();l++){
                    std::vector<double> sum_addends;
                    for(size_t m=0;m<current.w().nz();m++){
                        sum_addends.push_back(current.w().at(i,j,m)*((parent.v1()==current.order())?parent.w().at(m,k,l):parent.w().at(k,m,l)));
                    }
                    fused.at(i,j,k,l)=vec_add_float(sum_addends);
                }
            }
        }
    }
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
    size_t status=0;
    array3d<double> u;
    array1d<double> s;
    array3d<double> vt;
    svd(fused_mat,u,s,vt,status);
    for(size_t i=0;i<s.nx();i++){
        if(s.at(i)!=0){
            ee-=(s.at(i)*s.at(i))*log(s.at(i)*s.at(i));
        }
    }
    return ee;
}

size_t inner_svd_born(array3d<double>& fused_mat,array3d<double>& mat1,array3d<double>& mat2,size_t r_max,bool compress_r){
    size_t r;
    if(compress_r){
        r=1;
        size_t upper_bound_r_max=(fused_mat.nx()<fused_mat.ny())?fused_mat.nx():fused_mat.ny();
        while(r<=((upper_bound_r_max<r_max)?upper_bound_r_max:r_max)){
            mat1=array3d<double>(fused_mat.nx(),r,1);
            mat2=array3d<double>(r,fused_mat.ny(),1);
            double recon_err=truncated_svd(fused_mat,mat1,mat2,r); //make us vT, since current.w() must remain center and parent must remain isometric
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
        double recon_err=truncated_svd(fused_mat,mat1,mat2,r);
    }
    return r;
}

template<typename cmp>
void inner_updates_born(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
    z=calc_z_born(g);
    w=calc_w_born(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
    //update l/r env of parent and u_env of current
    // aux_update_lr_cache(current,parent,l_env_z,r_env_z,l_env_sample,r_env_sample);
    // aux_update_u_cache(current,parent,l_env_z,r_env_z,u_env_z,l_env_sample,r_env_sample,u_env_sample);
    
    // z=update_cache_z(g,current.order(),l_env_z,r_env_z,u_env_z);
    // w=update_cache_w(g,current.order(),l_env_sample,r_env_sample,u_env_sample);
    
    array3d<double> current_m(current.w().nx(),current.w().ny(),current.w().nz());
    array3d<double> current_v(current.w().nx(),current.w().ny(),current.w().nz());
    array3d<double> parent_m(parent.w().nx(),parent.w().ny(),parent.w().nz());
    array3d<double> parent_v(parent.w().nx(),parent.w().ny(),parent.w().nz());
    
    //single-site updates
    for(size_t t2=1;t2<=single_site_update_count;t2++){
        optimize::site_update_born(g,current,z,w,l_env_sample,r_env_sample,u_env_sample,current_m,current_v,t2,lr,beta1,beta2,epsilon);
        
        //qr decompose the weight tensor with lower legs fused, then push r matrix upwards
        array3d<double> target=matricize(current.w(),2);
        //compute qr
        array3d<double> q;
        array3d<double> r;
        size_t status=0;
        qr(target,q,r,status);
        current.w()=tensorize(q,current.w().nx(),current.w().ny(),2);
        if(current.order()==parent.v1()){
            array3d<double> res(r.nx(),parent.w().ny(),parent.w().nz());
            for(size_t i=0;i<r.nx();i++){
                for(size_t j=0;j<parent.w().ny();j++){
                    for(size_t k=0;k<parent.w().nz();k++){
                        std::vector<double> res_addends;
                        for(size_t l=0;l<r.ny();l++){
                            res_addends.push_back(parent.w().at(l,j,k)*r.at(i,l,0));
                        }
                        res.at(i,j,k)=vec_add_float(res_addends);
                    }
                }
            }
            parent.w()=res;
        }
        else{
            array3d<double> res(parent.w().nx(),r.nx(),parent.w().nz());
            for(size_t i=0;i<parent.w().nx();i++){
                for(size_t j=0;j<r.nx();j++){
                    for(size_t k=0;k<parent.w().nz();k++){
                        std::vector<double> res_addends;
                        for(size_t l=0;l<r.ny();l++){
                            res_addends.push_back(parent.w().at(i,l,k)*r.at(j,l,0));
                        }
                        res.at(i,j,k)=vec_add_float(res_addends);
                    }
                }
            }
            parent.w()=res;
        }
        
        normalize(parent.w()); //only normalize r
        
        g.vs()[current.order()].p_bond()=current;
        g.es().erase(it);
        it=g.es().insert(current);
        g.vs()[parent.order()].p_bond()=parent;
        g.es().erase(it_parent);
        it_parent=g.es().insert(parent);
        
        g.vs()[parent.order()].rank()=parent.w().nz();
        g.center_idx()=parent.order();
        
        z=calc_z_born(g);
        // w=calc_w_born(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        
        // update l/r env of parent
        aux_update_lr_cache_born(current,parent,l_env_sample,r_env_sample);
        aux_update_u_cache_born(current,parent,l_env_sample,r_env_sample,u_env_sample);

        w=update_cache_w_born(g,current.order(),l_env_sample,r_env_sample,u_env_sample);
        
        optimize::site_update_born(g,parent,z,w,l_env_sample,r_env_sample,u_env_sample,parent_m,parent_v,t2,lr,beta1,beta2,epsilon);
        
        //qr decompose the weight tensor with upper leg and other leg fused, then push r matrix left/rightwards
        if(current.order()==parent.v1()){
            array3d<double> target=matricize(parent.w(),0);
            //compute qr
            qr(target,q,r,status);
            parent.w()=tensorize(q,parent.w().ny(),parent.w().nz(),0);
        }
        else{
            array3d<double> target=matricize(parent.w(),1);
            //compute qr
            qr(target,q,r,status);
            parent.w()=tensorize(q,parent.w().nx(),parent.w().nz(),1);
        }
        array3d<double> res(current.w().nx(),current.w().ny(),r.nx());
        for(size_t i=0;i<current.w().nx();i++){
            for(size_t j=0;j<current.w().ny();j++){
                for(size_t k=0;k<r.nx();k++){
                    std::vector<double> res_addends;
                    for(size_t l=0;l<r.ny();l++){
                        res_addends.push_back(current.w().at(i,j,l)*r.at(k,l,0));
                    }
                    res.at(i,j,k)=vec_add_float(res_addends);
                }
            }
        }
        current.w()=res;
        
        normalize(current.w()); //only normalize r
        
        g.vs()[current.order()].p_bond()=current;
        g.es().erase(it);
        it=g.es().insert(current);
        g.vs()[parent.order()].p_bond()=parent;
        g.es().erase(it_parent);
        it_parent=g.es().insert(parent);
        
        g.vs()[current.order()].rank()=current.w().nz();
        g.center_idx()=current.order();
        
        z=calc_z_born(g);
        // w=calc_w_born(g,samples,labels,l_env_sample,r_env_sample,u_env_sample); //also calculate envs
        
        //update u_env of current
        aux_update_lr_cache_born(current,parent,l_env_sample,r_env_sample);
        aux_update_u_cache_born(current,parent,l_env_sample,r_env_sample,u_env_sample);
        
        w=update_cache_w_born(g,parent.order(),l_env_sample,r_env_sample,u_env_sample);
    }
}
template void inner_updates_born(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);

template<typename cmp>
double way1_born(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,array4d<double>& fused,size_t r_max,bool compress_r,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
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
    size_t r=inner_svd_born(fused_mat,mat1,mat2,r_max,compress_r);
    
    current.w()=array3d<double>(fused.nx(),fused.ny(),r);
    parent.w()=(parent.v1()==current.order())?array3d<double>(r,fused.nz(),fused.nw()):array3d<double>(fused.nz(),r,fused.nw());
    for(size_t i=0;i<mat1.nx();i++){
        for(size_t j=0;j<mat1.ny();j++){
            current.w().at(i/current.w().ny(),i%current.w().ny(),j)=mat1.at(i,j,0);
        }
    }
    for(size_t i=0;i<mat2.nx();i++){
        for(size_t j=0;j<mat2.ny();j++){
            if(parent.v1()==current.order()){
                parent.w().at(i,j/parent.w().nz(),j%parent.w().nz())=mat2.at(i,j,0);
            }
            else{
                parent.w().at(j/parent.w().nz(),i,j%parent.w().nz())=mat2.at(i,j,0);
            }
        }
    }
    
    g.vs()[current.order()].rank()=r;
    
    //qr decompose the weight tensor with upper leg and other leg fused, then push r matrix left/rightwards
    array3d<double> q_mat;
    array3d<double> r_mat;
    size_t status=0;
    if(current.order()==parent.v1()){
        array3d<double> target=matricize(parent.w(),0);
        //compute qr
        qr(target,q_mat,r_mat,status);
        parent.w()=tensorize(q_mat,parent.w().ny(),parent.w().nz(),0);
    }
    else{
        array3d<double> target=matricize(parent.w(),1);
        //compute qr
        qr(target,q_mat,r_mat,status);
        parent.w()=tensorize(q_mat,parent.w().nx(),parent.w().nz(),1);
    }
    array3d<double> res(current.w().nx(),current.w().ny(),r_mat.nx());
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<r_mat.nx();k++){
                std::vector<double> res_addends;
                for(size_t l=0;l<r_mat.ny();l++){
                    res_addends.push_back(current.w().at(i,j,l)*r_mat.at(k,l,0));
                }
                res.at(i,j,k)=vec_add_float(res_addends);
            }
        }
    }
    current.w()=res;
    
    g.vs()[current.order()].rank()=current.w().nz();
    g.center_idx()=current.order();
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    
    inner_updates_born(g,it,it_parent,current,parent,z,w,l_env_sample,r_env_sample,u_env_sample,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    double bmi=inner_bmi_born(g,current,parent,w,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way1: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    return bmi;
}
template double way1_born(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array4d<double>&,size_t,bool,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);

template<typename cmp>
double way2_born(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,array4d<double>& fused,size_t r_max,bool compress_r,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
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
    size_t r=inner_svd_born(fused_mat,mat1,mat2,r_max,compress_r);
    
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
            current.w().at(i/current.w().ny(),i%current.w().ny(),j)=mat1.at(i,j,0);
        }
    }
    for(size_t i=0;i<mat2.nx();i++){
        for(size_t j=0;j<mat2.ny();j++){
            if(parent.v1()==current.order()){
                parent.w().at(i,j/parent.w().nz(),j%parent.w().nz())=mat2.at(i,j,0);
            }
            else{
                parent.w().at(j/parent.w().nz(),i,j%parent.w().nz())=mat2.at(i,j,0);
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
    
    //qr decompose the weight tensor with upper leg and other leg fused, then push r matrix left/rightwards
    array3d<double> q_mat;
    array3d<double> r_mat;
    size_t status=0;
    if(current.order()==parent.v1()){
        array3d<double> target=matricize(parent.w(),0);
        //compute qr
        qr(target,q_mat,r_mat,status);
        parent.w()=tensorize(q_mat,parent.w().ny(),parent.w().nz(),0);
    }
    else{
        array3d<double> target=matricize(parent.w(),1);
        //compute qr
        qr(target,q_mat,r_mat,status);
        parent.w()=tensorize(q_mat,parent.w().nx(),parent.w().nz(),1);
    }
    array3d<double> res(current.w().nx(),current.w().ny(),r_mat.nx());
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<r_mat.nx();k++){
                std::vector<double> res_addends;
                for(size_t l=0;l<r_mat.ny();l++){
                    res_addends.push_back(current.w().at(i,j,l)*r_mat.at(k,l,0));
                }
                res.at(i,j,k)=vec_add_float(res_addends);
            }
        }
    }
    current.w()=res;
    
    g.center_idx()=current.order();
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    
    inner_updates_born(g,it,it_parent,current,parent,z,w,l_env_sample,r_env_sample,u_env_sample,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    bond center=g.vs()[g.center_idx()].p_bond();
    double bmi=inner_bmi_born(g,current,parent,w,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way2: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    return bmi;
}
template double way2_born(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array4d<double>&,size_t,bool,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);

template<typename cmp>
double way3_born(graph<cmp>& g,typename std::multiset<bond,cmp>::iterator& it,typename std::multiset<bond,cmp>::iterator& it_parent,bond& current,bond& parent,double& z,std::vector<double>& w,std::vector<std::vector<array1d<double> > >& l_env_sample,std::vector<std::vector<array1d<double> > >& r_env_sample,std::vector<std::vector<array1d<double> > >& u_env_sample,array4d<double>& fused,size_t r_max,bool compress_r,std::vector<sample_data>& samples,std::vector<size_t>& labels,size_t single_site_update_count,double lr,double beta1,double beta2,double epsilon){
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
    size_t r=inner_svd_born(fused_mat,mat1,mat2,r_max,compress_r);
    
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
            current.w().at(i/current.w().ny(),i%current.w().ny(),j)=mat1.at(i,j,0);
        }
    }
    for(size_t i=0;i<mat2.nx();i++){
        for(size_t j=0;j<mat2.ny();j++){
            if(parent.v1()==current.order()){
                parent.w().at(i,j/parent.w().nz(),j%parent.w().nz())=mat2.at(i,j,0);
            }
            else{
                parent.w().at(j/parent.w().nz(),i,j%parent.w().nz())=mat2.at(i,j,0);
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
    
    //qr decompose the weight tensor with upper leg and other leg fused, then push r matrix left/rightwards
    array3d<double> q_mat;
    array3d<double> r_mat;
    size_t status=0;
    if(current.order()==parent.v1()){
        array3d<double> target=matricize(parent.w(),0);
        //compute qr
        qr(target,q_mat,r_mat,status);
        parent.w()=tensorize(q_mat,parent.w().ny(),parent.w().nz(),0);
    }
    else{
        array3d<double> target=matricize(parent.w(),1);
        //compute qr
        qr(target,q_mat,r_mat,status);
        parent.w()=tensorize(q_mat,parent.w().nx(),parent.w().nz(),1);
    }
    array3d<double> res(current.w().nx(),current.w().ny(),r_mat.nx());
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<r_mat.nx();k++){
                std::vector<double> res_addends;
                for(size_t l=0;l<r_mat.ny();l++){
                    res_addends.push_back(current.w().at(i,j,l)*r_mat.at(k,l,0));
                }
                res.at(i,j,k)=vec_add_float(res_addends);
            }
        }
    }
    current.w()=res;
    
    g.center_idx()=current.order();
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    
    inner_updates_born(g,it,it_parent,current,parent,z,w,l_env_sample,r_env_sample,u_env_sample,samples,labels,single_site_update_count,lr,beta1,beta2,epsilon);
    
    bond center=g.vs()[g.center_idx()].p_bond();
    double bmi=inner_bmi_born(g,current,parent,w,l_env_sample,r_env_sample,u_env_sample);
    // std::cout<<"way3: "<<(std::string) g<<"\n";
    
    //update bmi and p_bond after every optimization sweep
    current.bmi()=bmi;
    g.vs()[current.order()].bmi()=bmi;
    
    g.vs()[current.order()].p_bond()=current;
    g.es().erase(it);
    it=g.es().insert(current);
    g.vs()[parent.order()].p_bond()=parent;
    g.es().erase(it_parent);
    it_parent=g.es().insert(parent);
    return bmi;
}
template double way3_born(graph<bmi_comparator>&,typename std::multiset<bond,bmi_comparator>::iterator&,typename std::multiset<bond,bmi_comparator>::iterator&,bond&,bond&,double&,std::vector<double>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,array4d<double>&,size_t,bool,std::vector<sample_data>&,std::vector<size_t>&,size_t,double,double,double,double);