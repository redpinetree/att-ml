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
            array3d<double> parent_w=parent.w();
            array3d<double> res(r.nx(),parent.w().ny(),parent.w().nz());
            #pragma omp parallel for collapse(3)
            for(int i=0;i<res.nx();i++){
                for(int j=0;j<res.ny();j++){
                    for(int k=0;k<res.nz();k++){
                        for(int l=0;l<r.ny();l++){
                            res.at(i,j,k)+=parent_w.at(l,j,k)*r.at(i,l,0);
                        }
                    }
                }
            }
            parent.w()=res;
        }
        else{
            array3d<double> parent_w=parent.w();
            array3d<double> res(parent.w().nx(),r.nx(),parent.w().nz());
            #pragma omp parallel for collapse(3)
            for(int i=0;i<res.nx();i++){
                for(int j=0;j<res.ny();j++){
                    for(int k=0;k<res.nz();k++){
                        for(int l=0;l<r.ny();l++){
                            res.at(i,j,k)+=parent_w.at(i,l,k)*r.at(j,l,0);
                        }
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
        array3d<double> current_w=current.w();
        array3d<double> res(current.w().nx(),current.w().ny(),r.nx());
        std::vector<double> res_addends(r.ny());
        #pragma omp parallel for collapse(3)
        for(int i=0;i<res.nx();i++){
            for(int j=0;j<res.ny();j++){
                for(int k=0;k<res.nz();k++){
                    for(int l=0;l<r.ny();l++){
                        res.at(i,j,k)+=current_w.at(i,j,l)*r.at(k,l,0);
                    }
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
    array3d<double> center_w=g.vs()[g.center_idx()].p_bond().w();
    double z=0;
    // #pragma omp parallel for reduction(+:z)
    for(int n=0;n<center_w.nx()*center_w.ny()*center_w.nz();n++){
        int k=n%center_w.nz();
        int j=(n/center_w.nz())%center_w.ny();
        int i=n/(center_w.ny()*center_w.nz());
        z+=center_w.at(i,j,k)*center_w.at(i,j,k);
    }
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
            w[s]=contracted_vectors[g.vs().size()-1].at(labels[s],s)*contracted_vectors[g.vs().size()-1].at(labels[s],s);
        }
        else{
            w[s]=0;
            for(int k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w[s]+=contracted_vectors[g.vs().size()-1].at(k,s)*contracted_vectors[g.vs().size()-1].at(k,s);
            }
        }
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
                std::vector<double> res_vec_r_addends(b_w.nx()*b_w.nz());
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
template std::vector<double> calc_w_born(graph<bmi_comparator>&,std::vector<std::vector<array1d<double> > >&,std::vector<int>&,std::vector<array2d<double> >&,std::vector<array2d<double> >&,std::vector<array2d<double> >&);


std::vector<double> update_cache_w_born(bond& current,bond& parent,std::vector<array2d<double> >& l_env,std::vector<array2d<double> >& r_env,std::vector<array2d<double> >& u_env){
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
            double psi=0;
            for(int k=0;k<current_w.nz();k++){
                psi+=l_env[parent_order].at(k,s)*u_env[current_order].at(k,s);
            }
            w[s]=psi*psi;
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
            double psi=0;
            for(int k=0;k<current_w.nz();k++){
                psi+=r_env[parent_order].at(k,s)*u_env[current_order].at(k,s);
            }
            w[s]=psi*psi;
            if(w[s]<1e-300){w[s]=1e-300;}
        }
    }
    return w;
}