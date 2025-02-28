#include <cstring>
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
                array1d<double> res_vec((*it).w().nz());
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

std::vector<std::vector<array3d<double> > > calc_dw(std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){ //calculate it for all tensors simultaneously
    std::vector<std::vector<array3d<double> > > res;
    res.reserve(u_env.size());
    for(int n=0;n<u_env.size();n++){
        std::vector<array3d<double> > batched_res_vec;
        batched_res_vec.reserve(u_env[n].ny());
        array3d<double> res_element(l_env[n].nx(),r_env[n].nx(),u_env[n].nx());
        #pragma omp parallel for firstprivate(res_element)
        for(int s=0;s<u_env[n].ny();s++){
            for(int i=0;i<l_env[n].nx();i++){
                for(int j=0;j<r_env[n].nx();j++){
                    for(int k=0;k<u_env[n].nx();k++){
                        res_element.at(i,j,k)=l_env[n].at(i,s)*r_env[n].at(j,s)*u_env[n].at(k,s);
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
std::vector<double> calc_w(graph<cmp>& g,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
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
                vec.at(labels[s],s)=1;
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
    
    int contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()].nx()!=0)&&(contracted_vectors[(*it).v2()].nx()!=0)&&((it==--g.es().end())||(contracted_vectors[(*it).order()].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                array2d<double> res_vec((*it).w().nz(),n_samples);
                std::vector<double> res_vec_addends(contracted_vectors[(*it).v1()].nx()*contracted_vectors[(*it).v2()].nx());
                #pragma omp parallel for firstprivate(res_vec_addends)
                for(int s=0;s<n_samples;s++){
                    for(int k=0;k<(*it).w().nz();k++){
                        size_t pos=0;
                        for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                            for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
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
            w[s]=contracted_vectors[g.vs().size()-1].at(labels[s],s);
        }
        else{
            std::vector<double> w_addends(g.vs()[g.vs().size()-1].rank());
            for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w_addends[k]=contracted_vectors[g.vs().size()-1].at(k,s);
            }
            w[s]=vec_add_float(w_addends);
        }
    }
    
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            if((u_env[(*it).v1()].nx()==0)&&(u_env[(*it).v2()].nx()==0)){
                if(u_env[(*it).v1()].nx()!=contracted_vectors[(*it).v1()].nx()){
                    u_env[(*it).v1()]=array2d<double>(contracted_vectors[(*it).v1()].nx(),u_env[(*it).v1()].ny());
                }
                std::vector<double> res_vec_l_addends(contracted_vectors[(*it).v2()].nx()*(*it).w().nz());
                #pragma omp parallel for firstprivate(res_vec_l_addends)
                for(int s=0;s<n_samples;s++){
                    for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        size_t pos=0;
                        for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_l_addends[pos]=r_env[(*it).order()].at(j,s)*u_env[(*it).order()].at(k,s)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        u_env[(*it).v1()].at(i,s)=vec_add_float(res_vec_l_addends);
                    }
                }
                if(u_env[(*it).v2()].nx()!=contracted_vectors[(*it).v2()].nx()){
                    u_env[(*it).v2()]=array2d<double>(contracted_vectors[(*it).v2()].nx(),u_env[(*it).v2()].ny());
                }
                std::vector<double> res_vec_r_addends(contracted_vectors[(*it).v1()].nx()*(*it).w().nz());
                #pragma omp parallel for firstprivate(res_vec_r_addends)
                for(int s=0;s<n_samples;s++){
                    for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                        size_t pos=0;
                        for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                            for(int k=0;k<(*it).w().nz();k++){
                                res_vec_r_addends[pos]=l_env[(*it).order()].at(i,s)*u_env[(*it).order()].at(k,s)*(*it).w().at(i,j,k);
                                pos++;
                            }
                        }
                        u_env[(*it).v2()].at(j,s)=vec_add_float(res_vec_r_addends);
                    }
                }
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    return w;
}
template std::vector<double> calc_w(graph<bmi_comparator>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);

template<typename cmp>
std::vector<double> update_cache_w(graph<cmp>& g,int center,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
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
            w_addends[k]=res_vec.at(k);
        }
        w[s]=vec_add_float(w_addends);
    }
    return w;
}
template std::vector<double> update_cache_w(graph<bmi_comparator>&,int,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);

//EXPERIMENTAL: only recompute Z using current and parent tensor + 4 env vectors
double update_cache_z(bond& current,bond& parent,std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){
    double z=0;
    array1d<double> res_vec(current.w().nz());
    std::vector<double> res_vec_addends(current.w().nx()*current.w().ny());
    array1d<double> res_vec2(current.w().nz());
    std::vector<double> res_vec2_addends((parent.v1()==current.order())?(parent.w().ny()*parent.w().nz()):(parent.w().nx()*parent.w().nz()));
    for(int k=0;k<current.w().nz();k++){
        size_t pos=0;
        for(int i=0;i<current.w().nx();i++){
            for(int j=0;j<current.w().ny();j++){
                res_vec_addends[pos]=l_env[current.order()].at(i)*r_env[current.order()].at(j)*current.w().at(i,j,k);
                pos++;
            }
        }
        res_vec.at(k)=vec_add_float(res_vec_addends);
    }
    if(parent.v1()==current.order()){
        l_env[parent.order()]=res_vec;
    }
    else{
        r_env[parent.order()]=res_vec;
    }
    
    if(parent.v1()==current.order()){
        for(int i=0;i<parent.w().nx();i++){
            size_t pos=0;
            for(int j=0;j<parent.w().ny();j++){
                for(int k=0;k<parent.w().nz();k++){
                    res_vec2_addends[pos]=r_env[parent.order()].at(j)*u_env[parent.order()].at(k)*parent.w().at(i,j,k);
                    pos++;
                }
            }
            res_vec2.at(i)=vec_add_float(res_vec2_addends);
        }
        u_env[current.order()]=res_vec2;
    }
    else{
        for(int j=0;j<parent.w().ny();j++){
            size_t pos=0;
            for(int i=0;i<parent.w().nx();i++){
                for(int k=0;k<parent.w().nz();k++){
                    res_vec2_addends[pos]=l_env[parent.order()].at(i)*u_env[parent.order()].at(k)*parent.w().at(i,j,k);
                    pos++;
                }
            }
            res_vec2.at(j)=vec_add_float(res_vec2_addends);
        }
        u_env[current.order()]=res_vec2;
    }
    
    
    std::vector<double> z_addends(current.w().nz());
    for(int k=0;k<current.w().nz();k++){
        z_addends[k]=res_vec.at(k)*res_vec2.at(k);
    }
    z=vec_add_float(z_addends);
    return z;
}

//EXPERIMENTAL: only recompute W using current and parent tensor + 4 env vectors
std::vector<double> update_cache_w(bond& current,bond& parent,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
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
        
        std::vector<double> w_addends(current.w().nz());
        for(int k=0;k<current.w().nz();k++){
            if(parent.v1()==current.order()){
                w_addends[k]=l_env[parent.order()].at(k,s)*u_env[current.order()].at(k,s);
            }
            else{
                w_addends[k]=r_env[parent.order()].at(k,s)*u_env[current.order()].at(k,s);
            }
        }
        w[s]=vec_add_float(w_addends);
    }
    return w;
}