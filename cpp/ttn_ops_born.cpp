#include <cstring>
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
std::vector<double> calc_w_born(graph<cmp>& g,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
    l_env.clear();
    r_env.clear();
    u_env.clear();
    std::vector<array2d<double> > contracted_vectors; //batched vectors
    contracted_vectors.reserve(g.vs().size());
    int n_samples=samples.size();
    for(int n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            array2d<double> vec(g.vs()[n].rank(),n_samples);
            #pragma omp parallel for
            for(int s=0;s<n_samples;s++){
                memcpy(&(vec.e())[g.vs()[n].rank()*s],&(samples[s][n].e())[0],g.vs()[n].rank()*sizeof(double));
            }
            contracted_vectors.push_back(vec);
        }
        else if((n==g.vs().size()-1)&&(labels.size()!=0)){ //top tensor
            array2d<double> vec(g.vs()[n].rank(),n_samples);
            #pragma omp parallel for
            for(int s=0;s<n_samples;s++){
                for(int a=0;a<vec.nx();a++){
                    if(a==labels[s]){ //if a==labels[s], element is 1. else 0
                        vec.at(a,s)=1;
                    }
                }
            }
            contracted_vectors.push_back(vec);
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(array2d<double>(0,n_samples));
        }
        l_env.push_back(array2d<double>(0,n_samples));
        r_env.push_back(array2d<double>(0,n_samples));
        u_env.push_back(array2d<double>(0,n_samples));
    }
    
    int contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()].nx()!=0)&&(contracted_vectors[(*it).v2()].nx()!=0)&&((it==--g.es().end())||(contracted_vectors[(*it).order()].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                array2d<double> res_vec((*it).w().nz(),n_samples);
                std::vector<double> res_vec_addends((*it).w().nx()*(*it).w().ny());
                #pragma omp parallel for firstprivate(res_vec_addends)
                for(int s=0;s<n_samples;s++){
                    for(int k=0;k<(*it).w().nz();k++){
                        size_t pos=0;
                        for(int i=0;i<(*it).w().nx();i++){
                            for(int j=0;j<(*it).w().ny();j++){
                                res_vec_addends[pos]=contracted_vectors[(*it).v1()].at(i,s)*contracted_vectors[(*it).v2()].at(j,s)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        res_vec.at(k,s)=vec_add_float(res_vec_addends);
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
            w[s]=contracted_vectors[g.vs().size()-1].at(labels[s],s)*contracted_vectors[g.vs().size()-1].at(labels[s],s);
        }
        else{
            std::vector<double> w_addends(g.vs()[g.vs().size()-1].rank());
            for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w_addends[k]=contracted_vectors[g.vs().size()-1].at(k,s)*contracted_vectors[g.vs().size()-1].at(k,s);
            }
            w[s]=vec_add_float(w_addends);
        }
    }
    
    if(labels.size()==0){ //top tensor's u_env is all ones
        u_env[g.vs().size()-1]=array2d<double>(g.vs()[g.vs().size()-1].rank(),n_samples);
        for(int s=0;s<n_samples;s++){
            for(int i=0;i<g.vs()[g.vs().size()-1].rank();i++){
                u_env[g.vs().size()-1].at(i,s)=1;
            }
        }
    }
    else{
        u_env[g.vs().size()-1]=contracted_vectors[g.vs().size()-1];
    }
    
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            if((u_env[(*it).v1()].nx()==0)&&(u_env[(*it).v2()].nx()==0)){
                array2d<double> res_vec_l((*it).w().nx(),n_samples);
                std::vector<double> res_vec_l_addends((*it).w().ny()*(*it).w().nz());
                #pragma omp parallel for firstprivate(res_vec_l_addends)
                for(int s=0;s<n_samples;s++){
                    for(int i=0;i<(*it).w().nx();i++){
                        size_t pos=0;
                        for(int j=0;j<(*it).w().ny();j++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_l_addends[pos]=r_env[(*it).order()].at(j,s)*u_env[(*it).order()].at(k,s)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        res_vec_l.at(i,s)=vec_add_float(res_vec_l_addends);
                    }
                }
                u_env[(*it).v1()]=res_vec_l;
                array2d<double> res_vec_r((*it).w().ny(),n_samples);
                std::vector<double> res_vec_r_addends((*it).w().nx()*(*it).w().nz());
                #pragma omp parallel for firstprivate(res_vec_r_addends)
                for(int s=0;s<n_samples;s++){
                    for(int j=0;j<(*it).w().ny();j++){
                        size_t pos=0;
                        for(int i=0;i<(*it).w().nx();i++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_r_addends[pos]=l_env[(*it).order()].at(i,s)*u_env[(*it).order()].at(k,s)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        res_vec_r.at(j,s)=vec_add_float(res_vec_r_addends);
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
template std::vector<double> calc_w_born(graph<bmi_comparator>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);

template<typename cmp>
std::vector<double> update_cache_w_born(graph<cmp>& g,int center,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
    int n_samples=u_env[g.vs().size()-1].ny();
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
                    if(u_env[(*it).v1()].nx()!=(*it).w().nx()){
                        u_env[(*it).v1()]=array2d<double>((*it).w().nx(),u_env[(*it).v1()].ny());
                    }
                    std::vector<double> res_vec_addends((*it).w().ny()*(*it).w().nz());
                    for(int i=0;i<(*it).w().nx();i++){
                        size_t pos=0;
                        for(int j=0;j<(*it).w().ny();j++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_addends[pos]=u_env[idx].at(k,s)*(*it).w().at(i,j,k);
                                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                                    res_vec_addends[pos]*=r_env[idx].at(j,s);
                                }
                                pos++;
                            }
                        }
                        u_env[(*it).v1()].at(i,s)=vec_add_float(res_vec_addends);
                    }
                    done_idxs.insert((*it).v1());
                }
            }
            {
                if(done_idxs.find((*it).v2())==done_idxs.end()){ //skip if finished
                    if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                        todo.push_back((*it).v2());
                    }
                    if(u_env[(*it).v1()].nx()!=(*it).w().ny()){
                        u_env[(*it).v1()]=array2d<double>((*it).w().ny(),u_env[(*it).v1()].ny());
                    }
                    std::vector<double> res_vec_addends((*it).w().nx()*(*it).w().nz());
                    for(int j=0;j<(*it).w().ny();j++){
                        size_t pos=0;
                        for(int i=0;i<(*it).w().nx();i++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_addends[pos]=u_env[idx].at(k,s)*(*it).w().at(i,j,k);
                                if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                                    res_vec_addends[pos]*=l_env[idx].at(i,s);
                                }
                                pos++;
                            }
                        }
                        u_env[(*it).v2()].at(j,s)=vec_add_float(res_vec_addends);
                    }
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
                if(g.vs()[u_idx].l_idx()==(*it2).order()){
                    if(l_env[u_idx].nx()!=(*it2).w().nz()){
                        l_env[u_idx]=array2d<double>((*it2).w().nz(),l_env[u_idx].ny());
                    }
                }
                else{
                    if(r_env[u_idx].nx()!=(*it2).w().nz()){
                        r_env[u_idx]=array2d<double>((*it2).w().nz(),r_env[u_idx].ny());
                    }
                }
                std::vector<double> res_vec_addends((*it2).w().nx()*(*it2).w().ny());
                for(int k=0;k<(*it2).w().nz();k++){
                    size_t pos=0;
                    for(int i=0;i<(*it2).w().nx();i++){
                        for(int j=0;j<(*it2).w().ny();j++){
                            res_vec_addends[pos]=l_env[(*it2).order()].at(i,s)*r_env[(*it2).order()].at(j,s)*(*it2).w().at(i,j,k);
                            pos++;
                        }
                    }
                    if(g.vs()[u_idx].l_idx()==(*it2).order()){
                        l_env[u_idx].at(k,s)=vec_add_float(res_vec_addends);
                    }
                    else{
                        r_env[u_idx].at(k,s)=vec_add_float(res_vec_addends);
                    }
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
                    res_vec_addends[pos]=l_env[(*it4).order()].at(i,s)*r_env[(*it4).order()].at(j,s)*(*it4).w().at(i,j,k);
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
template std::vector<double> update_cache_w_born(graph<bmi_comparator>&,int,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);

//EXPERIMENTAL: only recompute W using current and parent tensor + 4 env vectors
std::vector<double> update_cache_w_born(bond& current,bond& parent,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
    int n_samples=u_env[0].ny();
    std::vector<double> w(n_samples);
    std::vector<double> res_vec_addends(current.w().nx()*current.w().ny());
    std::vector<double> res_vec2_addends((parent.v1()==current.order())?(parent.w().ny()*parent.w().nz()):(parent.w().nx()*parent.w().nz()));
    #pragma omp parallel for firstprivate(res_vec_addends,res_vec2_addends)
    for(int s=0;s<n_samples;s++){
        if(parent.v1()==current.order()){
            if(l_env[parent.order()].nx()!=current.w().nz()){
                l_env[parent.order()]=array2d<double>(current.w().nz(),l_env[parent.order()].ny());
            }
        }
        else{
            if(r_env[parent.order()].nx()!=current.w().nz()){
                r_env[parent.order()]=array2d<double>(current.w().nz(),r_env[parent.order()].ny());
            }
        }
        for(int k=0;k<current.w().nz();k++){
            size_t pos=0;
            for(int i=0;i<current.w().nx();i++){
                for(int j=0;j<current.w().ny();j++){
                    res_vec_addends[pos]=l_env[current.order()].at(i,s)*r_env[current.order()].at(j,s)*current.w().at(i,j,k);
                    pos++;
                }
            }
            if(parent.v1()==current.order()){
                l_env[parent.order()].at(k,s)=vec_add_float(res_vec_addends);
            }
            else{
                r_env[parent.order()].at(k,s)=vec_add_float(res_vec_addends);
            }
        }
        
        if(parent.v1()==current.order()){
            if(u_env[current.order()].nx()!=parent.w().nx()){
                u_env[current.order()]=array2d<double>(parent.w().nx(),u_env[current.order()].ny());
            }
            for(int i=0;i<parent.w().nx();i++){
                size_t pos=0;
                for(int j=0;j<parent.w().ny();j++){
                    for(int k=0;k<parent.w().nz();k++){
                        res_vec2_addends[pos]=r_env[parent.order()].at(j,s)*u_env[parent.order()].at(k,s)*parent.w().at(i,j,k);
                        pos++;
                    }
                }
                u_env[current.order()].at(i,s)=vec_add_float(res_vec2_addends);
            }
        }
        else{
            if(u_env[current.order()].nx()!=parent.w().ny()){
                u_env[current.order()]=array2d<double>(parent.w().ny(),u_env[current.order()].ny());
            }
            for(int j=0;j<parent.w().ny();j++){
                size_t pos=0;
                for(int i=0;i<parent.w().nx();i++){
                    for(int k=0;k<parent.w().nz();k++){
                        res_vec2_addends[pos]=l_env[parent.order()].at(i,s)*u_env[parent.order()].at(k,s)*parent.w().at(i,j,k);
                        pos++;
                    }
                }
                u_env[current.order()].at(j,s)=vec_add_float(res_vec2_addends);
            }
        }
        
        std::vector<double> psi_addends(current.w().nz());
        for(int k=0;k<current.w().nz();k++){
            if(parent.v1()==current.order()){
                psi_addends[k]=l_env[parent.order()].at(k,s)*u_env[current.order()].at(k,s);
            }
            else{
                psi_addends[k]=r_env[parent.order()].at(k,s)*u_env[current.order()].at(k,s);
            }
        }
        double psi=vec_add_float(psi_addends);
        w[s]=psi*psi;
    }
    return w;
}