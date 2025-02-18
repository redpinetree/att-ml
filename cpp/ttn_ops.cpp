#include <deque>

#include "omp.h"

#include "mat_ops.hpp"
#include "ttn_ops.hpp"

void normalize(array3d<double>& w){
    std::vector<double> norm_addends(w.nx()*w.ny()*w.nz());
    #pragma omp parallel for collapse(3)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                norm_addends[(w.ny()*w.nx()*k)+(w.nx()*j)+i]=w.at(i,j,k)*w.at(i,j,k);
            }
        }
    }
    double norm=sqrt(vec_add_float(norm_addends));
    #pragma omp parallel for collapse(3)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                w.at(i,j,k)/=norm;
                if(!(fabs(w.at(i,j,k))>1e-100)){
                    // std::cout<<"replaced "<<w.at(i,j,k)<<"\n";
                    w.at(i,j,k)=1e-100;
                }
            }
        }
    }
}

void normalize(array4d<double>& w){
    std::vector<double> norm_addends(w.nx()*w.ny()*w.nz()*w.nw());
    #pragma omp parallel for collapse(4)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                for(int l=0;l<w.nw();l++){
                    norm_addends[(w.nz()*w.ny()*w.nx()*l)+(w.ny()*w.nx()*k)+(w.nx()*j)+i]=w.at(i,j,k,l)*w.at(i,j,k,l);
                }
            }
        }
    }
    double norm=sqrt(vec_add_float(norm_addends));
    #pragma omp parallel for collapse(4)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                for(int l=0;l<w.nw();l++){
                    w.at(i,j,k,l)/=norm;
                    if(!(fabs(w.at(i,j,k,l))>1e-100)){
                        // std::cout<<"replaced "<<w.at(i,j,k,l)<<"\n";
                        w.at(i,j,k,l)=1e-100;
                    }
                }
            }
        }
    }
}

void normalize_using_z(array3d<double>& w,double z){
    #pragma omp parallel for collapse(3)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                w.at(i,j,k)/=z;
                if(!(fabs(w.at(i,j,k))>1e-100)){
                    // std::cout<<"replaced "<<w.at(i,j,k)<<"\n";
                    w.at(i,j,k)=1e-100;
                }
            }
        }
    }
}

void normalize_using_z(array4d<double>& w,double z){
    #pragma omp parallel for collapse(4)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                for(int l=0;l<w.nw();l++){
                    w.at(i,j,k,l)/=z;
                    if(!(fabs(w.at(i,j,k,l))>1e-100)){
                        // std::cout<<"replaced "<<w.at(i,j,k,l)<<"\n";
                        w.at(i,j,k,l)=1e-100;
                    }
                }
            }
        }
    }
}

std::vector<array3d<double> > calc_dz(std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){ //calculate it for all tensors simultaneously
    std::vector<array3d<double> > res;
    res.reserve(u_env.size());
    for(int n=0;n<u_env.size();n++){
        array3d<double> res_element(l_env[n].nx(),r_env[n].nx(),u_env[n].nx());
        #pragma omp parallel for collapse(3)
        for(int i=0;i<l_env[n].nx();i++){
            for(int j=0;j<r_env[n].nx();j++){
                for(int k=0;k<u_env[n].nx();k++){
                    res_element.at(i,j,k)=l_env[n].at(i)*r_env[n].at(j)*u_env[n].at(k);
                }
            }
        }
        res.push_back(res_element);
    }
    return res;
}

template<typename cmp>
double calc_z(graph<cmp>& g){
    int contracted_idx_count=0;
    std::vector<array1d<double> > contracted_vectors;
    contracted_vectors.reserve(g.vs().size());
    for(int n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            array1d<double> v(g.vs()[n].rank());
            for(int i=0;i<v.nx();i++){
                v.at(i)=1;
            }
            contracted_vectors.push_back(v);
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
                for(int k=0;k<(*it).w().nz();k++){
                    std::vector<double> res_vec_addends(contracted_vectors[(*it).v1()].nx()*contracted_vectors[(*it).v2()].nx());
                    size_t pos=0;
                    for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                            res_vec_addends[pos]=contracted_vectors[(*it).v1()].at(i)*contracted_vectors[(*it).v2()].at(j)*(*it).w().at(i,j,k);
                            pos++;
                        }
                    }
                    res_vec.at(k)=vec_add_float(res_vec_addends);
                }
                contracted_vectors[(*it).order()]=res_vec;
                contracted_idx_count++;
                if(contracted_idx_count==g.vs().size()){break;}
            }
        }
    }
    std::vector<double> z_addends(g.vs()[g.vs().size()-1].rank());
    for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z_addends[k]=contracted_vectors[g.vs().size()-1].at(k);
    }
    double z=vec_add_float(z_addends);
    return z;
}
template double calc_z(graph<bmi_comparator>&);

template<typename cmp>
double calc_z(graph<cmp>& g,std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){
    l_env.clear();
    r_env.clear();
    u_env.clear();
    std::vector<array1d<double> > contracted_vectors;
    contracted_vectors.reserve(g.vs().size());
    for(int n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            array1d<double> v(g.vs()[n].rank());
            for(int i=0;i<v.nx();i++){
                v.at(i)=1;
            }
            contracted_vectors.push_back(v);
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(array1d<double>());
        }
        l_env.push_back(array1d<double>());
        r_env.push_back(array1d<double>());
        u_env.push_back(array1d<double>());
    }
    int contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()].nx()!=0)&&(contracted_vectors[(*it).v2()].nx()!=0)&&(contracted_vectors[(*it).order()].nx()==0)){ //process if children have been contracted and parent is not yet contracted
                array1d<double> res_vec(g.vs()[(*it).order()].rank());
                for(int k=0;k<(*it).w().nz();k++){
                    std::vector<double> res_vec_addends(contracted_vectors[(*it).v1()].nx()*contracted_vectors[(*it).v2()].nx());
                    size_t pos=0;
                    for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                            res_vec_addends[pos]=contracted_vectors[(*it).v1()].at(i)*contracted_vectors[(*it).v2()].at(j)*(*it).w().at(i,j,k);
                            pos++;
                        }
                    }
                    res_vec.at(k)=vec_add_float(res_vec_addends);
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
                if((*it).order()==g.vs().size()-1){ //top tensor's u_env is all ones
                    array1d<double> v(g.vs()[(*it).order()].rank());
                    for(int i=0;i<v.nx();i++){
                        v.at(i)=1;
                    }
                    u_env[(*it).order()]=v;
                }
                array1d<double> res_vec_l(g.vs()[(*it).v1()].rank());
                for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                    std::vector<double> res_vec_l_addends(contracted_vectors[(*it).v2()].nx()*(*it).w().nz());
                    size_t pos=0;
                    for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                        for(int k=0;k<(*it).w().nz();k++){
                            res_vec_l_addends[pos]=r_env[(*it).order()].at(j)*u_env[(*it).order()].at(k)*(*it).w().at(i,j,k);
                            pos++;
                        }
                    }
                    res_vec_l.at(i)=vec_add_float(res_vec_l_addends);
                }
                u_env[(*it).v1()]=res_vec_l;
                array1d<double> res_vec_r(g.vs()[(*it).v2()].rank());
                for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                    std::vector<double> res_vec_r_addends(contracted_vectors[(*it).v1()].nx()*(*it).w().nz());
                    size_t pos=0;
                    for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(int k=0;k<(*it).w().nz();k++){
                            res_vec_r_addends[pos]=l_env[(*it).order()].at(i)*u_env[(*it).order()].at(k)*(*it).w().at(i,j,k);
                            pos++;
                        }
                    }
                    res_vec_r.at(j)=vec_add_float(res_vec_r_addends);
                }
                u_env[(*it).v2()]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    std::vector<double> z_addends(g.vs()[g.vs().size()-1].rank());
    for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z_addends[k]=contracted_vectors[g.vs().size()-1].at(k);
    }
    double z=vec_add_float(z_addends);
    return z;
}
template double calc_z(graph<bmi_comparator>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);

template<typename cmp>
double update_cache_z(graph<cmp>& g,int center,std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){
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
                for(int i=0;i<(*it).w().nx();i++){
                    std::vector<double> res_vec_addends((*it).w().ny()*(*it).w().nz());
                    size_t pos=0;
                    for(int j=0;j<(*it).w().ny();j++){
                        for(int k=0;k<(*it).w().nz();k++){
                            res_vec_addends[pos]=u_env[idx].at(k)*(*it).w().at(i,j,k);
                            if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                                res_vec_addends[pos]*=r_env[idx].at(j);
                            }
                            pos++;
                        }
                    }
                    res_vec.at(i)=vec_add_float(res_vec_addends);
                }
                u_env[(*it).v1()]=res_vec;
                done_idxs.insert((*it).v1());
            }
        }
        {
            if(done_idxs.find((*it).v2())==done_idxs.end()){ //skip if finished
                if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                    todo.push_back((*it).v2());
                }
                array1d<double> res_vec((*it).w().ny());
                for(int j=0;j<(*it).w().ny();j++){
                    std::vector<double> res_vec_addends((*it).w().nx()*(*it).w().nz());
                    size_t pos=0;
                    for(int i=0;i<(*it).w().nx();i++){
                        for(int k=0;k<(*it).w().nz();k++){
                            res_vec_addends[pos]=u_env[idx].at(k)*(*it).w().at(i,j,k);
                            if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                                res_vec_addends[pos]*=l_env[idx].at(i);
                            }
                            pos++;
                        }
                    }
                    res_vec.at(j)=vec_add_float(res_vec_addends);
                }
                u_env[(*it).v2()]=res_vec;
                done_idxs.insert((*it).v2());
            }
        }
    }
    //update l_env and r_env of all upstream sites and note positions of untouched branches
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
            for(int k=0;k<(*it2).w().nz();k++){
                std::vector<double> res_vec_addends((*it2).w().nx()*(*it2).w().ny());
                size_t pos=0;
                for(int i=0;i<(*it2).w().nx();i++){
                    for(int j=0;j<(*it2).w().ny();j++){
                        res_vec_addends[pos]=l_env[(*it2).order()].at(i)*r_env[(*it2).order()].at(j)*(*it2).w().at(i,j,k);
                        pos++;
                    }
                }
                res_vec.at(k)=vec_add_float(res_vec_addends);
            }
            if(g.vs()[u_idx].l_idx()==(*it2).order()){
                l_env[u_idx]=res_vec;
            }
            else{
                r_env[u_idx]=res_vec;
            }
            todo.push_back(u_idx);
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
    for(int k=0;k<(*it4).w().nz();k++){
        std::vector<double> res_vec_addends((*it4).w().nx()*(*it4).w().ny());
        size_t pos=0;
        for(int i=0;i<(*it4).w().nx();i++){
            for(int j=0;j<(*it4).w().ny();j++){
                res_vec_addends[pos]=l_env[(*it4).order()].at(i)*r_env[(*it4).order()].at(j)*(*it4).w().at(i,j,k);
                pos++;
            }
        }
        res_vec.at(k)=vec_add_float(res_vec_addends);
    }
    std::vector<double> z_addends(g.vs()[top_idx].rank());
    for(int k=0;k<g.vs()[top_idx].rank();k++){
        z_addends[k]=res_vec.at(k);
    }
    double z=vec_add_float(z_addends);
    return z;
}
template double update_cache_z(graph<bmi_comparator>&,int,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);

std::vector<std::vector<array3d<double> > > calc_dw(std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){ //calculate it for all tensors simultaneously
    std::vector<std::vector<array3d<double> > > res;
    res.reserve(u_env.size());
    for(int n=0;n<u_env.size();n++){
        std::vector<array3d<double> > batched_res_vec;
        batched_res_vec.reserve(u_env[0].size());
        array3d<double> res_element(l_env[n][0].nx(),r_env[n][0].nx(),u_env[n][0].nx());
        #pragma omp parallel for firstprivate(res_element)
        for(int s=0;s<u_env[0].size();s++){
            for(int i=0;i<l_env[n][s].nx();i++){
                for(int j=0;j<r_env[n][s].nx();j++){
                    for(int k=0;k<u_env[n][s].nx();k++){
                        res_element.at(i,j,k)=l_env[n][s].at(i)*r_env[n][s].at(j)*u_env[n][s].at(k);
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
std::vector<double> calc_w(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<int>& labels,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
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
            #pragma omp parallel for
            for(int s=0;s<n_samples;s++){
                array1d<double> vec_e(g.vs()[n].rank());
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
            w[s]=contracted_vectors[g.vs().size()-1][s].at(labels[s]);
        }
        else{
            std::vector<double> w_addends(g.vs()[g.vs().size()-1].rank());
            for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w_addends[k]=contracted_vectors[g.vs().size()-1][s].at(k);
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
template std::vector<double> calc_w(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<int>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

template<typename cmp>
std::vector<double> update_cache_w(graph<cmp>& g,int center,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
    int n_samples=u_env[g.vs().size()-1].size();
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
                #pragma omp parallel for firstprivate(res_vec,res_vec_addends)
                for(int s=0;s<n_samples;s++){
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
                }
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
                #pragma omp parallel for firstprivate(res_vec,res_vec_addends)
                for(int s=0;s<n_samples;s++){
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
            array1d<double> res_vec(g.vs()[(*it2).order()].rank());
            std::vector<double> res_vec_addends((*it2).w().nx()*(*it2).w().ny());
            #pragma omp parallel for firstprivate(res_vec,res_vec_addends)
            for(int s=0;s<n_samples;s++){
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
    
    std::vector<double> w(n_samples);
    auto it4=g.es().rbegin();
    array1d<double> res_vec(g.vs()[(*it4).order()].rank());
    std::vector<double> res_vec_addends((*it4).w().nx()*(*it4).w().ny());
    #pragma omp parallel for firstprivate(res_vec,res_vec_addends)
    for(int s=0;s<n_samples;s++){
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
            w_addends[k]=res_vec.at(k);
        }
        w[s]=vec_add_float(w_addends);
    }
    return w;
}
template std::vector<double> update_cache_w(graph<bmi_comparator>&,int,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);