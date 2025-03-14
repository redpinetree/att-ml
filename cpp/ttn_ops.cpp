#include <cstring>
#include <deque>

#include "omp.h"

#include "mat_ops.hpp"
#include "ttn_ops.hpp"

void normalize(array3d<double>& w){
    double norm_sq=0;
    // #pragma omp parallel for collapse(3) reduction(+:norm_sq)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                norm_sq+=w.at(i,j,k)*w.at(i,j,k);
            }
        }
    }
    double norm=sqrt(norm_sq);
    #pragma omp parallel for collapse(3)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                w.at(i,j,k)/=norm;
                if(!(fabs(w.at(i,j,k))>1e-16)){
                    // std::cout<<"replaced "<<w.at(i,j,k)<<"\n";
                    w.at(i,j,k)=1e-16;
                }
            }
        }
    }
}

void normalize(array4d<double>& w){
    double norm_sq=0;
    // #pragma omp parallel for collapse(4) reduction(+:norm_sq)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                for(int l=0;l<w.nw();l++){
                    norm_sq+=w.at(i,j,k,l)*w.at(i,j,k,l);
                }
            }
        }
    }
    double norm=sqrt(norm_sq);
    #pragma omp parallel for collapse(4)
    for(int i=0;i<w.nx();i++){
        for(int j=0;j<w.ny();j++){
            for(int k=0;k<w.nz();k++){
                for(int l=0;l<w.nw();l++){
                    w.at(i,j,k,l)/=norm;
                    if(!(fabs(w.at(i,j,k,l))>1e-16)){
                        // std::cout<<"replaced "<<w.at(i,j,k,l)<<"\n";
                        w.at(i,j,k,l)=1e-16;
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
                if(!(fabs(w.at(i,j,k))>1e-16)){
                    // std::cout<<"replaced "<<w.at(i,j,k)<<"\n";
                    w.at(i,j,k)=1e-16;
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
                    if(!(fabs(w.at(i,j,k,l))>1e-16)){
                        // std::cout<<"replaced "<<w.at(i,j,k,l)<<"\n";
                        w.at(i,j,k,l)=1e-16;
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
                    for(int i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(int j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                            res_vec.at(k)+=contracted_vectors[(*it).v1()].at(i)*contracted_vectors[(*it).v2()].at(j)*(*it).w().at(i,j,k);
                        }
                    }
                }
                contracted_vectors[(*it).order()]=res_vec;
                contracted_idx_count++;
                if(contracted_idx_count==g.vs().size()){break;}
            }
        }
    }
    double z=0;
    for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z+=contracted_vectors[g.vs().size()-1].at(k);
    }
    if(z>1e300){z=1e300;}
    if(z<1e-300){z=1e-300;}
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
            int b_order=(*it).order();
            int b_v1=(*it).v1();
            int b_v2=(*it).v2();
            array3d<double> b_w=(*it).w();
            if((contracted_vectors[b_v1].nx()!=0)&&(contracted_vectors[b_v2].nx()!=0)&&(contracted_vectors[b_order].nx()==0)){ //process if children have been contracted and parent is not yet contracted
                array1d<double> res_vec(b_w.nz());
                for(int k=0;k<b_w.nz();k++){
                    for(int i=0;i<contracted_vectors[b_v1].nx();i++){
                        for(int j=0;j<contracted_vectors[b_v2].nx();j++){
                            res_vec.at(k)+=contracted_vectors[b_v1].at(i)*contracted_vectors[b_v2].at(j)*b_w.at(i,j,k);
                        }
                    }
                }
                contracted_vectors[b_order]=res_vec;
                l_env[b_order]=contracted_vectors[b_v1];
                r_env[b_order]=contracted_vectors[b_v2];
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            int b_order=(*it).order();
            int b_v1=(*it).v1();
            int b_v2=(*it).v2();
            array3d<double> b_w=(*it).w();
            if((u_env[b_v1].nx()==0)&&(u_env[b_v2].nx()==0)){
                if(b_order==g.vs().size()-1){ //top tensor's u_env is all ones
                    array1d<double> v(g.vs()[b_order].rank());
                    for(int i=0;i<v.nx();i++){
                        v.at(i)=1;
                    }
                    u_env[b_order]=v;
                }
                array1d<double> res_vec_l(g.vs()[b_v1].rank());
                for(int i=0;i<contracted_vectors[b_v1].nx();i++){
                    for(int j=0;j<contracted_vectors[b_v2].nx();j++){
                        for(int k=0;k<b_w.nz();k++){
                            res_vec_l.at(i)+=r_env[b_order].at(j)*u_env[b_order].at(k)*b_w.at(i,j,k);
                        }
                    }
                }
                u_env[b_v1]=res_vec_l;
                array1d<double> res_vec_r(g.vs()[b_v2].rank());
                for(int j=0;j<contracted_vectors[b_v2].nx();j++){
                    for(int i=0;i<contracted_vectors[b_v1].nx();i++){
                        for(int k=0;k<b_w.nz();k++){
                            res_vec_r.at(j)+=l_env[b_order].at(i)*u_env[b_order].at(k)*b_w.at(i,j,k);
                        }
                    }
                }
                u_env[b_v2]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    double z=0;
    for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z+=contracted_vectors[g.vs().size()-1].at(k);
    }
    if(z>1e300){z=1e300;}
    if(z<1e-300){z=1e-300;}
    return z;
}
template double calc_z(graph<bmi_comparator>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);

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
std::vector<double> calc_w(graph<cmp>& g,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels){
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
        else if((n==g.vs().size()-1)&&(labels.size()!=0)){ //top tensor
            array2d<double> vec(rank,n_samples);
            #pragma omp parallel for
            for(int s=0;s<n_samples;s++){
                vec.at(labels[s],s)=1;
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
            int b_order=(*it).order();
            int b_v1=(*it).v1();
            int b_v2=(*it).v2();
            array3d<double> b_w=(*it).w();
            if((contracted_vectors[b_v1].nx()!=0)&&(contracted_vectors[b_v2].nx()!=0)&&((it==--g.es().end())||(contracted_vectors[b_order].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                array2d<double> res_vec(b_w.nz(),n_samples);
                #pragma omp parallel for collapse(2)
                for(int s=0;s<n_samples;s++){
                    for(int k=0;k<b_w.nz();k++){
                        for(int i=0;i<b_w.nx();i++){
                            for(int j=0;j<b_w.ny();j++){
                                res_vec.at(k,s)+=contracted_vectors[b_v1].at(i,s)*contracted_vectors[b_v2].at(j,s)*b_w.at(i,j,k);
                            }
                        }
                    }
                }
                contracted_vectors[b_order]=res_vec;
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
            w[s]=0;
            for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w[s]+=contracted_vectors[g.vs().size()-1].at(k,s);
            }
        }
        if(w[s]>1e300){w[s]=1e300;}
        if(w[s]<1e-300){w[s]=1e-300;}
    }
    return w;
}
template std::vector<double> calc_w(graph<bmi_comparator>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&);

template<typename cmp>
std::vector<double> calc_w(graph<cmp>& g,std::vector<std::vector<array1d<double> > >& samples,std::vector<int>& labels,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
    l_env.clear();
    r_env.clear();
    u_env.clear();
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
        else if((n==g.vs().size()-1)&&(labels.size()!=0)){ //top tensor
            array2d<double> vec(rank,n_samples);
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
            int b_order=(*it).order();
            int b_v1=(*it).v1();
            int b_v2=(*it).v2();
            array3d<double> b_w=(*it).w();
            if((contracted_vectors[b_v1].nx()!=0)&&(contracted_vectors[b_v2].nx()!=0)&&((it==--g.es().end())||(contracted_vectors[b_order].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                array2d<double> res_vec(b_w.nz(),n_samples);
                #pragma omp parallel for collapse(2)
                for(int s=0;s<n_samples;s++){
                    for(int k=0;k<b_w.nz();k++){
                        for(int i=0;i<b_w.nx();i++){
                            for(int j=0;j<b_w.ny();j++){
                                res_vec.at(k,s)+=contracted_vectors[b_v1].at(i,s)*contracted_vectors[b_v2].at(j,s)*b_w.at(i,j,k);
                            }
                        }
                    }
                }
                contracted_vectors[b_order]=res_vec;
                l_env[b_order]=contracted_vectors[b_v1];
                r_env[b_order]=contracted_vectors[b_v2];
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
            w[s]=0;
            for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w[s]+=contracted_vectors[g.vs().size()-1].at(k,s);
            }
        }
        if(w[s]>1e300){w[s]=1e300;}
        if(w[s]<1e-300){w[s]=1e-300;}
    }
    
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            int b_order=(*it).order();
            int b_v1=(*it).v1();
            int b_v2=(*it).v2();
            array3d<double> b_w=(*it).w();
            if((u_env[b_v1].nx()==0)&&(u_env[b_v2].nx()==0)){
                array2d<double> res_vec_l(b_w.nx(),n_samples);
                #pragma omp parallel for collapse(2)
                for(int s=0;s<n_samples;s++){
                    for(int i=0;i<b_w.nx();i++){
                        for(int j=0;j<b_w.ny();j++){
                            for(int k=0;k<b_w.nz();k++){
                                res_vec_l.at(i,s)+=r_env[b_order].at(j,s)*u_env[b_order].at(k,s)*b_w.at(i,j,k);
                            }
                        }
                    }
                }
                u_env[b_v1]=res_vec_l;
                array2d<double> res_vec_r(b_w.ny(),n_samples);
                #pragma omp parallel for collapse(2)
                for(int s=0;s<n_samples;s++){
                    for(int j=0;j<b_w.ny();j++){
                        for(int i=0;i<b_w.nx();i++){
                            for(int k=0;k<b_w.nz();k++){
                                res_vec_r.at(j,s)+=l_env[b_order].at(i,s)*u_env[b_order].at(k,s)*b_w.at(i,j,k);
                            }
                        }
                    }
                }
                u_env[b_v2]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    return w;
}
template std::vector<double> calc_w(graph<bmi_comparator>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);

double update_cache_z(bond& current,bond& parent,std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){
    int current_order=current.order();
    int parent_order=parent.order();
    array3d<double> current_w=current.w();
    array3d<double> parent_w=parent.w();
    array1d<double> res_vec(current.w().nz());
    array1d<double> res_vec2(current.w().nz());
    for(int k=0;k<current_w.nz();k++){
        for(int i=0;i<current_w.nx();i++){
            for(int j=0;j<current_w.ny();j++){
                res_vec.at(k)+=l_env[current_order].at(i)*r_env[current_order].at(j)*current_w.at(i,j,k);
            }
        }
    }
    
    if(parent.v1()==current_order){
        l_env[parent_order]=res_vec;
        for(int i=0;i<parent_w.nx();i++){
            for(int j=0;j<parent_w.ny();j++){
                for(int k=0;k<parent_w.nz();k++){
                    res_vec2.at(i)+=r_env[parent_order].at(j)*u_env[parent_order].at(k)*parent_w.at(i,j,k);
                }
            }
        }
    }
    else{
        r_env[parent_order]=res_vec;
        for(int j=0;j<parent_w.ny();j++){
            for(int i=0;i<parent_w.nx();i++){
                for(int k=0;k<parent_w.nz();k++){
                    res_vec2.at(j)+=l_env[parent_order].at(i)*u_env[parent_order].at(k)*parent_w.at(i,j,k);
                }
            }
        }
    }
    u_env[current_order]=res_vec2;
    
    double z=0;
    for(int k=0;k<current_w.nz();k++){
        z+=res_vec.at(k)*res_vec2.at(k);
    }
    return z;
}

std::vector<double> update_cache_w(bond& current,bond& parent,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
    int n_samples=u_env[0].ny();
    int current_order=current.order();
    int parent_order=parent.order();
    array3d<double> current_w=current.w();
    array3d<double> parent_w=parent.w();
    std::vector<double> w(n_samples);
    if(parent.v1()==current_order){
        l_env[parent_order]=array2d<double>(current_w.nz(),l_env[parent_order].ny());
        u_env[current_order]=array2d<double>(parent_w.nx(),u_env[current_order].ny());
        #pragma omp parallel for collapse(2)
        for(int s=0;s<n_samples;s++){
            for(int k=0;k<current_w.nz();k++){
                for(int i=0;i<current_w.nx();i++){
                    for(int j=0;j<current_w.ny();j++){
                        l_env[parent_order].at(k,s)+=l_env[current_order].at(i,s)*r_env[current_order].at(j,s)*current_w.at(i,j,k);
                    }
                }
            }
        }
        #pragma omp parallel for collapse(2)
        for(int s=0;s<n_samples;s++){
            for(int i=0;i<parent_w.nx();i++){
                for(int j=0;j<parent_w.ny();j++){
                    for(int k=0;k<parent_w.nz();k++){
                        u_env[current_order].at(i,s)+=r_env[parent_order].at(j,s)*u_env[parent_order].at(k,s)*parent_w.at(i,j,k);
                    }
                }
            }
        }
        #pragma omp parallel for
        for(int s=0;s<n_samples;s++){
            for(int k=0;k<current_w.nz();k++){
                 w[s]+=l_env[parent_order].at(k,s)*u_env[current_order].at(k,s);
            }
            if(w[s]>1e300){w[s]=1e300;}
            if(w[s]<1e-300){w[s]=1e-300;}
        }
    }
    else{
        r_env[parent_order]=array2d<double>(current_w.nz(),r_env[parent_order].ny());
        u_env[current_order]=array2d<double>(parent_w.ny(),u_env[current_order].ny());
        #pragma omp parallel for collapse(2)
        for(int s=0;s<n_samples;s++){
            for(int k=0;k<current_w.nz();k++){
                for(int i=0;i<current_w.nx();i++){
                    for(int j=0;j<current_w.ny();j++){
                        r_env[parent_order].at(k,s)+=l_env[current_order].at(i,s)*r_env[current_order].at(j,s)*current_w.at(i,j,k);
                    }
                }
            }
        }
        #pragma omp parallel for collapse(2)
        for(int s=0;s<n_samples;s++){
            for(int j=0;j<parent_w.ny();j++){
                for(int i=0;i<parent_w.nx();i++){
                    for(int k=0;k<parent_w.nz();k++){
                        u_env[current_order].at(j,s)+=l_env[parent_order].at(i,s)*u_env[parent_order].at(k,s)*parent_w.at(i,j,k);
                    }
                }
            }
        }
        #pragma omp parallel for
        for(int s=0;s<n_samples;s++){
            for(int k=0;k<current_w.nz();k++){
                 w[s]+=r_env[parent_order].at(k,s)*u_env[current_order].at(k,s);
            }
            if(w[s]>1e300){w[s]=1e300;}
            if(w[s]<1e-300){w[s]=1e-300;}
        }
    }
    return w;
}