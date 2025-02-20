#include <deque>

#include "omp.h"

#include "mat_ops.hpp"
#include "ttn_ops.hpp"

template<typename cmp>
void canonicalize(graph<cmp>& g,int center_idx){
    if(center_idx<g.n_phys_sites()){
        std::cout<<"Center bond cannot be an input bond.\n";
        exit(1);
    }
    //canonicalize with center at the top tensor
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        // std::cout<<(std::string)current<<" ";
        // std::cout<<(std::string)g.vs()[current.order()].p_bond()<<"\n";
        if(current.order()==g.vs().size()-1){ //handle top tensor
            g.vs()[current.order()].p_bond()=current;
            g.es().erase(it);
            it=g.es().insert(current);
            break;
        }
        bond key;
        key.todo()=0;
        key.order()=g.vs()[(*it).order()].u_idx();
        key.depth()=g.vs()[key.order()].depth();
        key.bmi()=-1e50;
        auto it_parent=g.es().lower_bound(key);
        bond parent=*it_parent; //must be a dereferenced pointer to the actual object, not a copy!
        // std::cout<<(std::string) current<<"->"<<(std::string) parent<<"\n";
        
        //qr decompose the weight tensor with lower legs fused, then push r matrix upwards
        array3d<double> target=matricize(current.w(),2);
        //compute qr
        array3d<double> q;
        array3d<double> r;
        int status=0;
        qr(target,q,r,status);
        //set q and push r
        // if((q.nx()!=current.w().nx())||(q.ny()!=current.w().ny())){ //nothing happens if separated axis had dim 1
            current.w()=tensorize(q,current.w().nx(),current.w().ny(),2);
        // }
        if(current.order()==parent.v1()){
            array3d<double> res(r.nx(),parent.w().ny(),parent.w().nz());
            std::vector<double> res_addends(r.ny());
            #pragma omp parallel for collapse(3) firstprivate(res_addends)
            for(int i=0;i<r.nx();i++){
                for(int j=0;j<parent.w().ny();j++){
                    for(int k=0;k<parent.w().nz();k++){
                        for(int l=0;l<r.ny();l++){
                            res_addends[l]=parent.w().at(l,j,k)*r.at(i,l,0);
                        }
                        res.at(i,j,k)=vec_add_float(res_addends);
                    }
                }
            }
            parent.w()=res;
        }
        else{
            array3d<double> res(parent.w().nx(),r.nx(),parent.w().nz());
            std::vector<double> res_addends(r.ny());
            #pragma omp parallel for collapse(3) firstprivate(res_addends)
            for(int i=0;i<parent.w().nx();i++){
                for(int j=0;j<r.nx();j++){
                    for(int k=0;k<parent.w().nz();k++){
                        for(int l=0;l<r.ny();l++){
                            res_addends[l]=parent.w().at(i,l,k)*r.at(j,l,0);
                        }
                        res.at(i,j,k)=vec_add_float(res_addends);
                    }
                }
            }
            parent.w()=res;
        }
        
        normalize(parent.w());
        
        g.vs()[current.order()].p_bond()=current;
        g.es().erase(it);
        it=g.es().insert(current);
        g.vs()[parent.order()].p_bond()=parent;
        g.es().erase(it_parent);
        it_parent=g.es().insert(parent);
    }
    
    //make an edge the center of orthogonality
    // if(center_idx==g.vs().size()-1){
        // g.center_idx()=center_idx;
        // return;
    // }
    std::deque<int> todo;
    int idx=center_idx;
    while(idx!=g.vs().size()-1){
        todo.push_back(idx);
        idx=g.vs()[idx].u_idx();
    }
    idx=g.vs().size()-1;
    while(!todo.empty()){
        bond key;
        key.todo()=0;
        key.order()=idx;
        key.depth()=g.vs()[key.order()].depth();
        key.bmi()=-1e50;
        auto it_parent=g.es().lower_bound(key);
        bond parent=*it_parent;
        int next_idx=todo.back();
        todo.pop_back();
        key.order()=next_idx;
        key.depth()=g.vs()[key.order()].depth();
        auto it=g.es().lower_bound(key);
        bond current=*it;
        // std::cout<<(std::string) parent<<"->"<<(std::string) current<<"\n";
        
        //qr decompose the weight tensor with upper leg and other leg fused, then push r matrix left/rightwards
        array3d<double> q;
        array3d<double> r;
        int status=0;
        if(current.order()==parent.v1()){
            // std::cout<<"L: \n";
            array3d<double> target=matricize(parent.w(),0);
            //compute qr
            qr(target,q,r,status);
            //set q and push r
            parent.w()=tensorize(q,parent.w().ny(),parent.w().nz(),0);
        }
        else{
            // std::cout<<"R: \n";
            array3d<double> target=matricize(parent.w(),1);
            //compute qr
            qr(target,q,r,status);
            //set q and push r
            parent.w()=tensorize(q,parent.w().nx(),parent.w().nz(),1);
        }
        array3d<double> res(current.w().nx(),current.w().ny(),r.nx());
        std::vector<double> res_addends(r.ny());
        #pragma omp parallel for collapse(3) firstprivate(res_addends)
        for(int i=0;i<current.w().nx();i++){
            for(int j=0;j<current.w().ny();j++){
                for(int k=0;k<r.nx();k++){
                    for(int l=0;l<r.ny();l++){
                        res_addends[l]=current.w().at(i,j,l)*r.at(k,l,0);
                    }
                    res.at(i,j,k)=vec_add_float(res_addends);
                }
            }
        }
        current.w()=res;
        
        normalize(current.w());
        
        g.vs()[current.order()].p_bond()=current;
        g.es().erase(it);
        it=g.es().insert(current);
        g.vs()[parent.order()].p_bond()=parent;
        g.es().erase(it_parent);
        it_parent=g.es().insert(parent);
        
        g.vs()[current.order()].rank()=current.w().nz();
        g.vs()[parent.order()].rank()=parent.w().nz();
        
        idx=next_idx;
    }
    auto it_center=g.es().find(g.vs()[center_idx].p_bond());
    bond center=*it_center;
    g.es().erase(it_center);
    it_center=g.es().insert(center);
    g.center_idx()=center_idx;
}
template void canonicalize(graph<bmi_comparator>&,int);

template<typename cmp>
void canonicalize(graph<cmp>& g){
    canonicalize(g,g.vs().size()-1);
}
template void canonicalize(graph<bmi_comparator>&);

template<typename cmp>
double calc_z_born(graph<cmp>& g){
    bond center=g.vs()[g.center_idx()].p_bond();
    std::vector<double> z_addends(center.w().nx()*center.w().ny()*center.w().nz());
    #pragma omp parallel for collapse(3)
    for(int i=0;i<center.w().nx();i++){
        for(int j=0;j<center.w().ny();j++){
            for(int k=0;k<center.w().nz();k++){
                z_addends[(center.w().ny()*center.w().nx()*k)+(center.w().nx()*j)+i]=center.w().at(i,j,k)*center.w().at(i,j,k);
            }
        }
    }
    double z=vec_add_float(z_addends);
    return z;
}
template double calc_z_born(graph<bmi_comparator>&);

template<typename cmp>
std::vector<double> calc_w_born(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<int>& labels,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
    l_env.clear();
    r_env.clear();
    u_env.clear();
    std::vector<std::vector<array1d<double> > > contracted_vectors; //batched vectors
    contracted_vectors.reserve(g.vs().size());
    int n_samples=samples.size();
    for(int n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            std::vector<array1d<double> > vec(n_samples);
            #pragma omp parallel for
            for(int s=0;s<n_samples;s++){
                array1d<double> vec_e(g.vs()[n].rank());
                if(samples[s].s()[n]!=0){
                    for(int a=0;a<vec_e.nx();a++){
                        if(a==(samples[s].s()[n]-1)){ //if a==samples[s].s()[n]-1, element is 1. else 0
                            vec_e.at(a)=1;
                        }
                    }
                }
                vec[s]=vec_e;
            }
            contracted_vectors.push_back(vec);
        }
        else if((n==g.vs().size()-1)&&(labels.size()!=0)){ //top tensor
            std::vector<array1d<double> > vec(n_samples);
            array1d<double> vec_e(g.vs()[n].rank());
            #pragma omp parallel for
            for(int s=0;s<n_samples;s++){
                for(int a=0;a<vec_e.nx();a++){
                    if(a==labels[s]){ //if a==labels[s], element is 1. else 0
                        vec_e.at(a)=1;
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
    
    int contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()][0].nx()!=0)&&(contracted_vectors[(*it).v2()][0].nx()!=0)&&((it==--g.es().end())||(contracted_vectors[(*it).order()][0].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                std::vector<array1d<double> > res_vec(n_samples,array1d<double>(g.vs()[(*it).order()].rank()));
                std::vector<double> res_vec_addends(contracted_vectors[(*it).v1()][0].nx()*contracted_vectors[(*it).v2()][0].nx());
                #pragma omp parallel for firstprivate(res_vec_addends)
                for(int s=0;s<n_samples;s++){
                    for(int k=0;k<(*it).w().nz();k++){
                        size_t pos=0;
                        for(int i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                            for(int j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                                res_vec_addends[pos]=contracted_vectors[(*it).v1()][s].at(i)*contracted_vectors[(*it).v2()][s].at(j)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        res_vec[s].at(k)=vec_add_float(res_vec_addends);
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
    
    std::vector<double> w(n_samples);
    #pragma omp parallel for
    for(int s=0;s<n_samples;s++){
        if(labels.size()!=0){
            w[s]=contracted_vectors[g.vs().size()-1][s].at(labels[s])*contracted_vectors[g.vs().size()-1][s].at(labels[s]);
        }
        else{
            std::vector<double> w_addends(g.vs()[g.vs().size()-1].rank());
            for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w_addends[k]=contracted_vectors[g.vs().size()-1][s].at(k)*contracted_vectors[g.vs().size()-1][s].at(k);
            }
            w[s]=vec_add_float(w_addends);
        }
    }
    
    if(labels.size()==0){ //top tensor's u_env is all ones
        u_env[g.vs().size()-1]=std::vector<array1d<double> >(n_samples);
        for(int s=0;s<n_samples;s++){
            array1d<double> v(g.vs()[g.vs().size()-1].rank());
            for(int i=0;i<v.nx();i++){
                v.at(i)=1;
            }
            u_env[g.vs().size()-1][s]=v;
        }
    }
    else{
        u_env[g.vs().size()-1]=contracted_vectors[g.vs().size()-1];
    }
    
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            if((u_env[(*it).v1()][0].nx()==0)&&(u_env[(*it).v2()][0].nx()==0)){
                std::vector<array1d<double> > res_vec_l(n_samples,array1d<double>(g.vs()[(*it).v1()].rank()));
                std::vector<double> res_vec_l_addends(contracted_vectors[(*it).v2()][0].nx()*(*it).w().nz());
                #pragma omp parallel for firstprivate(res_vec_l_addends)
                for(int s=0;s<n_samples;s++){
                    for(int i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                        size_t pos=0;
                        for(int j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_l_addends[pos]=r_env[(*it).order()][s].at(j)*u_env[(*it).order()][s].at(k)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        res_vec_l[s].at(i)=vec_add_float(res_vec_l_addends);
                    }
                }
                u_env[(*it).v1()]=res_vec_l;
                std::vector<array1d<double> > res_vec_r(n_samples,array1d<double>(g.vs()[(*it).v2()].rank()));
                std::vector<double> res_vec_r_addends(contracted_vectors[(*it).v1()][0].nx()*(*it).w().nz());
                #pragma omp parallel for firstprivate(res_vec_r_addends)
                for(int s=0;s<n_samples;s++){
                    for(int j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                        size_t pos=0;
                        for(int i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_r_addends[pos]=l_env[(*it).order()][s].at(i)*u_env[(*it).order()][s].at(k)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        res_vec_r[s].at(j)=vec_add_float(res_vec_r_addends);
                    }
                }
                u_env[(*it).v2()]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    return w;
}
template std::vector<double> calc_w_born(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<int>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

template<typename cmp>
std::vector<double> update_cache_w_born(graph<cmp>& g,int center,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
    int n_samples=u_env[g.vs().size()-1].size();
    std::vector<double> w(n_samples);
    #pragma omp parallel for
    for(int s=0;s<n_samples;s++){
        std::deque<int> todo;
        std::set<int> done_idxs;
        todo.push_back(center);
        //update u_env of all downstream sites
        while(!todo.empty()){ //iterate over multiset repeatedly until all idxs processed
            int idx=todo.front();
            todo.pop_front();
            bond key;
            key.todo()=0;
            key.order()=idx;
            key.depth()=g.vs()[idx].depth();
            key.bmi()=-1e50;
            auto it=g.es().lower_bound(key);
            //each time, update the u_env of the left and right child
            {
                if(done_idxs.find((*it).v1())==done_idxs.end()){ //skip if finished
                    if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                        todo.push_back((*it).v1());
                    }
                    array1d<double> res_vec((*it).w().nx());
                    std::vector<double> res_vec_addends((*it).w().ny()*(*it).w().nz());
                    for(int i=0;i<(*it).w().nx();i++){
                        size_t pos=0;
                        for(int j=0;j<(*it).w().ny();j++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_addends[pos]=u_env[idx][s].at(k)*(*it).w().at(i,j,k);
                                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                                    res_vec_addends[pos]*=r_env[idx][s].at(j);
                                }
                                pos++;
                            }
                        }
                        res_vec.at(i)=vec_add_float(res_vec_addends);
                    }
                    u_env[(*it).v1()][s]=res_vec;
                    done_idxs.insert((*it).v1());
                }
            }
            {
                if(done_idxs.find((*it).v2())==done_idxs.end()){ //skip if finished
                    if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                        todo.push_back((*it).v2());
                    }
                    array1d<double> res_vec((*it).w().ny());
                    std::vector<double> res_vec_addends((*it).w().nx()*(*it).w().nz());
                    for(int j=0;j<(*it).w().ny();j++){
                        size_t pos=0;
                        for(int i=0;i<(*it).w().nx();i++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_addends[pos]=u_env[idx][s].at(k)*(*it).w().at(i,j,k);
                                if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                                    res_vec_addends[pos]*=l_env[idx][s].at(i);
                                }
                                pos++;
                            }
                        }
                        res_vec.at(j)=vec_add_float(res_vec_addends);
                    }
                    u_env[(*it).v2()][s]=res_vec;
                    done_idxs.insert((*it).v2());
                }
            }
        }
        //update l_env and r_env of all upstream sites
        bond key;
        key.todo()=0;
        key.order()=center;
        key.depth()=g.vs()[center].depth();
        key.bmi()=-1e50;
        auto it2=g.es().lower_bound(key);
        bond center_bond=*it2;
        int top_idx=g.vs().size()-1;
        if((*it2).order()!=top_idx){
            while(1){ // each time, update the l_env or r_env of the current tensor
                int u_idx=g.vs()[(*it2).order()].u_idx();
                array1d<double> res_vec(g.vs()[(*it2).order()].rank());
                std::vector<double> res_vec_addends((*it2).w().nx()*(*it2).w().ny());
                for(int k=0;k<(*it2).w().nz();k++){
                    size_t pos=0;
                    for(int i=0;i<(*it2).w().nx();i++){
                        for(int j=0;j<(*it2).w().ny();j++){
                            res_vec_addends[pos]=l_env[(*it2).order()][s].at(i)*r_env[(*it2).order()][s].at(j)*(*it2).w().at(i,j,k);
                            pos++;
                        }
                    }
                    res_vec.at(k)=vec_add_float(res_vec_addends);
                }
                if(g.vs()[u_idx].l_idx()==(*it2).order()){
                    l_env[u_idx][s]=res_vec;
                }
                else{
                    r_env[u_idx][s]=res_vec;
                }
                done_idxs.insert((*it2).order());
                if(u_idx==top_idx){
                    break;
                }
                bond key;
                key.todo()=0;
                key.order()=u_idx;
                key.depth()=g.vs()[u_idx].depth();
                key.bmi()=-1e50;
                it2=g.es().lower_bound(key);
            }
        }
        
        auto it4=g.es().rbegin();
        array1d<double> res_vec(g.vs()[(*it4).order()].rank());
        std::vector<double> res_vec_addends((*it4).w().nx()*(*it4).w().ny());
        for(int k=0;k<(*it4).w().nz();k++){
            size_t pos=0;
            for(int i=0;i<(*it4).w().nx();i++){
                for(int j=0;j<(*it4).w().ny();j++){
                    res_vec_addends[pos]=l_env[(*it4).order()][s].at(i)*r_env[(*it4).order()][s].at(j)*(*it4).w().at(i,j,k);
                    pos++;
                }
            }
            res_vec.at(k)=vec_add_float(res_vec_addends);
        }
        std::vector<double> w_addends(g.vs()[top_idx].rank());
        for(int k=0;k<g.vs()[top_idx].rank();k++){
            w_addends[k]=res_vec.at(k)*res_vec.at(k);
        }
        w[s]=vec_add_float(w_addends);
    }
    return w;
}
template std::vector<double> update_cache_w_born(graph<bmi_comparator>&,int,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

//EXPERIMENTAL: only recompute W using current and parent tensor + 4 env vectors
std::vector<double> update_cache_w_born(bond& current,bond& parent,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
    int n_samples=u_env[0].size();
    std::vector<double> w(n_samples);
    array1d<double> res_vec(current.w().nz());
    std::vector<double> res_vec_addends(current.w().nx()*current.w().ny());
    array1d<double> res_vec2(current.w().nz());
    std::vector<double> res_vec2_addends((parent.v1()==current.order())?(parent.w().ny()*parent.w().nz()):(parent.w().nx()*parent.w().nz()));
    #pragma omp parallel for firstprivate(res_vec,res_vec_addends,res_vec2,res_vec2_addends)
    for(int s=0;s<n_samples;s++){
        for(int k=0;k<current.w().nz();k++){
            size_t pos=0;
            for(int i=0;i<current.w().nx();i++){
                for(int j=0;j<current.w().ny();j++){
                    res_vec_addends[pos]=l_env[current.order()][s].at(i)*r_env[current.order()][s].at(j)*current.w().at(i,j,k);
                    pos++;
                }
            }
            res_vec.at(k)=vec_add_float(res_vec_addends);
        }
        if(parent.v1()==current.order()){
            l_env[parent.order()][s]=res_vec;
        }
        else{
            r_env[parent.order()][s]=res_vec;
        }
        
        if(parent.v1()==current.order()){
            for(int i=0;i<parent.w().nx();i++){
                size_t pos=0;
                for(int j=0;j<parent.w().ny();j++){
                    for(int k=0;k<parent.w().nz();k++){
                        res_vec2_addends[pos]=r_env[parent.order()][s].at(j)*u_env[parent.order()][s].at(k)*parent.w().at(i,j,k);
                        pos++;
                    }
                }
                res_vec2.at(i)=vec_add_float(res_vec2_addends);
            }
        }
        else{
            for(int j=0;j<parent.w().ny();j++){
                size_t pos=0;
                for(int i=0;i<parent.w().nx();i++){
                    for(int k=0;k<parent.w().nz();k++){
                        res_vec2_addends[pos]=l_env[parent.order()][s].at(i)*u_env[parent.order()][s].at(k)*parent.w().at(i,j,k);
                        pos++;
                    }
                }
                res_vec2.at(j)=vec_add_float(res_vec2_addends);
            }
        }
        u_env[current.order()][s]=res_vec2;
        
        std::vector<double> psi_addends(current.w().nz());
        for(int k=0;k<current.w().nz();k++){
            psi_addends[k]=res_vec.at(k)*res_vec2.at(k);
        }
        double psi=vec_add_float(psi_addends);
        w[s]=psi*psi;
    }
    return w;
}