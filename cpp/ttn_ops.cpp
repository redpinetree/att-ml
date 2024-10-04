#include <deque>

#include "ttn_ops.hpp"

std::vector<array3d<double> > calc_dz(std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){ //calculate it for all tensors simultaneously
    std::vector<array3d<double> > res;
    for(size_t n=0;n<u_env.size();n++){
        array3d<double> res_element(l_env[n].nx(),r_env[n].nx(),u_env[n].nx());
        for(size_t i=0;i<l_env[n].nx();i++){
            for(size_t j=0;j<r_env[n].nx();j++){
                for(size_t k=0;k<u_env[n].nx();k++){
                    res_element.at(i,j,k)=l_env[n].at(i)+r_env[n].at(j)+u_env[n].at(k); //log space
                }
            }
        }
        res.push_back(res_element);
    }
    return res;
}

template<typename cmp>
double calc_z(graph<cmp>& g){
    size_t contracted_idx_count=0;
    std::vector<array1d<double> > contracted_vectors;
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            contracted_vectors.push_back(array1d<double>(g.vs()[n].rank())); //0 because log(1)=0
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
                for(size_t k=0;k<(*it).w().nz();k++){
                    std::vector<double> res_vec_addends;
                    for(size_t i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(size_t j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                            res_vec_addends.push_back(contracted_vectors[(*it).v1()].at(i)+contracted_vectors[(*it).v2()].at(j)+(*it).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec.at(k)=lse(res_vec_addends); //log space
                }
                contracted_vectors[(*it).order()]=res_vec;
                contracted_idx_count++;
                if(contracted_idx_count==g.vs().size()){break;}
            }
        }
    }
    std::vector<double> z_addends;
    for(size_t k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z_addends.push_back(contracted_vectors[g.vs().size()-1].at(k));
    }
    double z=lse(z_addends);
    return z;
}
template double calc_z(graph<bmi_comparator>&);

template<typename cmp>
double calc_z(graph<cmp>& g,std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env){
    l_env.clear();
    r_env.clear();
    u_env.clear();
    std::vector<array1d<double> > contracted_vectors;
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            contracted_vectors.push_back(array1d<double>(g.vs()[n].rank())); //0 because log(1)=0
        }
        else{ //virtual sites correspond to tensors
            contracted_vectors.push_back(array1d<double>());
        }
        l_env.push_back(array1d<double>());
        r_env.push_back(array1d<double>());
        u_env.push_back(array1d<double>());
    }
    size_t contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()].nx()!=0)&&(contracted_vectors[(*it).v2()].nx()!=0)&&(contracted_vectors[(*it).order()].nx()==0)){ //process if children have been contracted and parent is not yet contracted
                array1d<double> res_vec(g.vs()[(*it).order()].rank());
                for(size_t k=0;k<(*it).w().nz();k++){
                    std::vector<double> res_vec_addends;
                    for(size_t i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(size_t j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                            res_vec_addends.push_back(contracted_vectors[(*it).v1()].at(i)+contracted_vectors[(*it).v2()].at(j)+(*it).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec.at(k)=lse(res_vec_addends); //log space
                }
                contracted_vectors[(*it).order()]=res_vec;
                l_env[(*it).order()]=contracted_vectors[(*it).v1()];
                r_env[(*it).order()]=contracted_vectors[(*it).v2()];
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    // std::cout<<"l_env:\n:";
    // for(size_t a=0;a<l_env.size();a++){
        // std::cout<<(std::string)l_env[a]<<"\n";
    // }
    // std::cout<<"r_env:\n:";
    // for(size_t a=0;a<r_env.size();a++){
        // std::cout<<(std::string)r_env[a]<<"\n";
    // }
    // std::cout<<"u_env:\n:";
    // for(size_t a=0;a<u_env.size();a++){
        // std::cout<<(std::string)u_env[a]<<"\n";
    // }
    contracted_idx_count=0; //reset counter
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset, in reverse, repeatedly until all idxs processed
        for(auto it=g.es().rbegin();it!=g.es().rend();++it){
            if((u_env[(*it).v1()].nx()==0)&&(u_env[(*it).v2()].nx()==0)){
                if((*it).order()==g.vs().size()-1){ //top tensor's u_env is all ones
                    u_env[(*it).order()]=array1d<double>(g.vs()[(*it).order()].rank()); //0 because log(1)=0
                }
                array1d<double> res_vec_l(g.vs()[(*it).v1()].rank());
                for(size_t i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                    std::vector<double> res_vec_l_addends;
                    for(size_t j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            res_vec_l_addends.push_back(r_env[(*it).order()].at(j)+u_env[(*it).order()].at(k)+(*it).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec_l.at(i)=lse(res_vec_l_addends); //log space
                }
                u_env[(*it).v1()]=res_vec_l;
                array1d<double> res_vec_r(g.vs()[(*it).v2()].rank());
                for(size_t j=0;j<contracted_vectors[(*it).v2()].nx();j++){
                    std::vector<double> res_vec_r_addends;
                    for(size_t i=0;i<contracted_vectors[(*it).v1()].nx();i++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            res_vec_r_addends.push_back(l_env[(*it).order()].at(i)+u_env[(*it).order()].at(k)+(*it).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec_r.at(j)=lse(res_vec_r_addends); //log space
                }
                u_env[(*it).v2()]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    std::vector<double> z_addends;
    for(size_t k=0;k<g.vs()[g.vs().size()-1].rank();k++){
        z_addends.push_back(contracted_vectors[g.vs().size()-1].at(k));
    }
    double z=lse(z_addends);
    return z;
}
template double calc_z(graph<bmi_comparator>&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&);

template<typename cmp>
double update_cache_z(graph<cmp>& g,size_t center,std::vector<array1d<double> >& l_env,std::vector<array1d<double> >& r_env,std::vector<array1d<double> >& u_env,std::set<size_t>& todo_keys){
    std::deque<size_t> todo;
    std::set<size_t> done_idxs;
    todo.push_back(center);
    //update u_env of all downstream sites
    while(!todo.empty()){ //iterate over multiset repeatedly until all idxs processed
        size_t idx=todo.front();
        todo.pop_front();
        bond key;
        key.todo()=0;
        key.order()=idx;
        key.depth()=g.vs()[idx].depth();
        key.bmi()=-1e50;
        key.virt_count()=2;
        auto it=g.es().lower_bound(key);
        //each time, update the u_env of the left and right child
        {
            if(done_idxs.find((*it).v1())==done_idxs.end()){ //skip if finished
                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                    todo.push_back((*it).v1());
                }
                array1d<double> res_vec((*it).w().nx());
                for(size_t i=0;i<(*it).w().nx();i++){
                    std::vector<double> res_vec_addends;
                    for(size_t j=0;j<(*it).w().ny();j++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                                res_vec_addends.push_back(r_env[idx].at(j)+u_env[idx].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                            else{
                                res_vec_addends.push_back(u_env[idx].at(k)+(*it).w().at(i,j,k));
                            }
                        }
                    }
                    res_vec.at(i)=lse(res_vec_addends); //log space
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
                for(size_t j=0;j<(*it).w().ny();j++){
                    std::vector<double> res_vec_addends;
                    for(size_t i=0;i<(*it).w().nx();i++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                                res_vec_addends.push_back(l_env[idx].at(i)+u_env[idx].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                            else{
                                res_vec_addends.push_back(u_env[idx].at(k)+(*it).w().at(i,j,k));
                            }
                        }
                    }
                    res_vec.at(j)=lse(res_vec_addends); //log space
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
    key.virt_count()=2;
    auto it2=g.es().lower_bound(key);
    bond center_bond=*it2;
    size_t top_idx=g.vs().size()-1;
    if((*it2).order()!=top_idx){
        while(1){ // each time, update the l_env or r_env of the current tensor
            size_t u_idx=g.vs()[(*it2).order()].u_idx();
            array1d<double> res_vec(g.vs()[(*it2).order()].rank());
            for(size_t k=0;k<(*it2).w().nz();k++){
                std::vector<double> res_vec_addends;
                for(size_t i=0;i<(*it2).w().nx();i++){
                    for(size_t j=0;j<(*it2).w().ny();j++){
                        res_vec_addends.push_back(l_env[(*it2).order()].at(i)+r_env[(*it2).order()].at(j)+(*it2).w().at(i,j,k)); //log space
                    }
                }
                res_vec.at(k)=lse(res_vec_addends); //log space
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
            key.virt_count()=2;
            it2=g.es().lower_bound(key);
        }
    }
    while(!todo.empty()){ //iterate over multiset repeatedly until all idxs processed, skipping finished branches
        size_t idx=todo.front();
        todo.pop_front();
        bond key;
        key.todo()=0;
        key.order()=idx;
        key.depth()=g.vs()[idx].depth();
        key.bmi()=-1e50;
        key.virt_count()=2;
        auto it=g.es().lower_bound(key);
        //each time, update the u_env of the left and right child
        {
            if(done_idxs.find((*it).v1())==done_idxs.end()){ //skip if finished
                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                    todo.push_back((*it).v1());
                }
                array1d<double> res_vec((*it).w().nx());
                for(size_t i=0;i<(*it).w().nx();i++){
                    std::vector<double> res_vec_addends;
                    for(size_t j=0;j<(*it).w().ny();j++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                                res_vec_addends.push_back(r_env[idx].at(j)+u_env[idx].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                            else{
                                res_vec_addends.push_back(u_env[idx].at(k)+(*it).w().at(i,j,k));
                            }
                        }
                    }
                    res_vec.at(i)=lse(res_vec_addends); //log space
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
                for(size_t j=0;j<(*it).w().ny();j++){
                    std::vector<double> res_vec_addends;
                    for(size_t i=0;i<(*it).w().nx();i++){
                        for(size_t k=0;k<(*it).w().nz();k++){
                            if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                                res_vec_addends.push_back(l_env[idx].at(i)+u_env[idx].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                            else{
                                res_vec_addends.push_back(u_env[idx].at(k)+(*it).w().at(i,j,k));
                            }
                        }
                    }
                    res_vec.at(j)=lse(res_vec_addends); //log space
                }
                u_env[(*it).v2()]=res_vec;
                done_idxs.insert((*it).v2());
            }
        }
    }
    std::vector<double> z_addends;
    auto it4=g.es().rbegin();
    array1d<double> res_vec(g.vs()[(*it4).order()].rank());
    for(size_t k=0;k<(*it4).w().nz();k++){
        std::vector<double> res_vec_addends;
        for(size_t i=0;i<(*it4).w().nx();i++){
            for(size_t j=0;j<(*it4).w().ny();j++){
                res_vec_addends.push_back(l_env[(*it4).order()].at(i)+r_env[(*it4).order()].at(j)+(*it4).w().at(i,j,k)); //log space
            }
        }
        res_vec.at(k)=lse(res_vec_addends); //log space
    }
    for(size_t k=0;k<g.vs()[top_idx].rank();k++){
        z_addends.push_back(res_vec.at(k));
    }
    double z=lse(z_addends);
    return z;
}
template double update_cache_z(graph<bmi_comparator>&,size_t,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::vector<array1d<double> >&,std::set<size_t>&);

std::vector<std::vector<array3d<double> > > calc_dw(std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){ //calculate it for all tensors simultaneously
    std::vector<std::vector<array3d<double> > > res;
    for(size_t n=0;n<u_env.size();n++){
        std::vector<array3d<double> > batched_res_vec;
        for(size_t s=0;s<u_env[0].size();s++){
            array3d<double> res_element(l_env[n][s].nx(),r_env[n][s].nx(),u_env[n][s].nx());
            for(size_t i=0;i<l_env[n][s].nx();i++){
                for(size_t j=0;j<r_env[n][s].nx();j++){
                    for(size_t k=0;k<u_env[n][s].nx();k++){
                        res_element.at(i,j,k)=l_env[n][s].at(i)+r_env[n][s].at(j)+u_env[n][s].at(k); //log space
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
std::vector<double> calc_w(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<size_t>& labels,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
    l_env.clear();
    r_env.clear();
    u_env.clear();
    std::vector<std::vector<array1d<double> > > contracted_vectors; //batched vectors
    size_t n_samples=samples.size();
    for(size_t n=0;n<g.vs().size();n++){
        if(n<g.n_phys_sites()){ //physical (input) sites do not correspond to tensors, so environment is empty vector
            std::vector<array1d<double> > vec(n_samples);
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
        else if((n==g.vs().size()-1)&&(labels.size()!=0)){ //top tensor
            std::vector<array1d<double> > vec(n_samples);
            for(size_t s=0;s<n_samples;s++){
                array1d<double> vec_e(g.vs()[n].rank());
                for(size_t a=0;a<vec_e.nx();a++){
                    if(a!=labels[s]){ //if a==labels[s], element is log(1)=0. else log(0)=-inf
                        vec_e.at(a)=log(1e-100);
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
    
    if(labels.size()==0){ //top tensor's u_env is all ones
        u_env[g.vs().size()-1]=std::vector<array1d<double> >(n_samples,array1d<double>(g.vs()[g.vs().size()-1].rank())); //0 because log(1)=0
    }
    else{
        u_env[g.vs().size()-1]=contracted_vectors[g.vs().size()-1];
    }
    
    size_t contracted_idx_count=0;
    while(contracted_idx_count!=(g.vs().size()-g.n_phys_sites())){ //iterate over multiset repeatedly until all idxs processed
        for(auto it=g.es().begin();it!=g.es().end();++it){
            if((contracted_vectors[(*it).v1()][0].nx()!=0)&&(contracted_vectors[(*it).v2()][0].nx()!=0)&&((it==--g.es().end())||(contracted_vectors[(*it).order()][0].nx()==0))){ //process if children have been contracted and (parent is not yet contracted OR is top)
                std::vector<array1d<double> > res_vec(n_samples,array1d<double>(g.vs()[(*it).order()].rank()));
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
            if((u_env[(*it).v1()][0].nx()==0)&&(u_env[(*it).v2()][0].nx()==0)){
                std::vector<array1d<double> > res_vec_l(n_samples,array1d<double>(g.vs()[(*it).v1()].rank()));
                for(size_t s=0;s<n_samples;s++){
                    for(size_t i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                        std::vector<double> res_vec_l_addends;
                        for(size_t j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                res_vec_l_addends.push_back(r_env[(*it).order()][s].at(j)+u_env[(*it).order()][s].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                        }
                        res_vec_l[s].at(i)=lse(res_vec_l_addends); //log space
                    }
                }
                u_env[(*it).v1()]=res_vec_l;
                std::vector<array1d<double> > res_vec_r(n_samples,array1d<double>(g.vs()[(*it).v2()].rank()));
                for(size_t s=0;s<n_samples;s++){
                    for(size_t j=0;j<contracted_vectors[(*it).v2()][s].nx();j++){
                        std::vector<double> res_vec_r_addends;
                        for(size_t i=0;i<contracted_vectors[(*it).v1()][s].nx();i++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                res_vec_r_addends.push_back(l_env[(*it).order()][s].at(i)+u_env[(*it).order()][s].at(k)+(*it).w().at(i,j,k)); //log space
                            }
                        }
                        res_vec_r[s].at(j)=lse(res_vec_r_addends); //log space
                    }
                }
                u_env[(*it).v2()]=res_vec_r;
                contracted_idx_count++;
                if(contracted_idx_count==(g.vs().size()-g.n_phys_sites())){break;}
            }
        }
    }
    std::vector<double> w(n_samples);
    for(size_t s=0;s<n_samples;s++){
        if(labels.size()!=0){
            w[s]=contracted_vectors[g.vs().size()-1][s].at(labels[s]);
        }
        else{
            std::vector<double> w_addends;
            for(size_t k=0;k<g.vs()[g.vs().size()-1].rank();k++){
                w_addends.push_back(contracted_vectors[g.vs().size()-1][s].at(k));
            }
            w[s]=lse(w_addends);
        }
    }
    return w;
}
template std::vector<double> calc_w(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<size_t>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

template<typename cmp>
std::vector<double> calc_w(graph<cmp>& g,std::vector<sample_data>& samples,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env){
    std::vector<size_t> dummy_labels;
    return calc_w(g,samples,dummy_labels,l_env,r_env,u_env);
}
template std::vector<double> calc_w(graph<bmi_comparator>&,std::vector<sample_data>&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&);

template<typename cmp>
std::vector<double> update_cache_w(graph<cmp>& g,size_t center,std::vector<std::vector<array1d<double> > >& l_env,std::vector<std::vector<array1d<double> > >& r_env,std::vector<std::vector<array1d<double> > >& u_env,std::set<size_t>& todo_keys){
    size_t n_samples=u_env[g.vs().size()-1].size();
    std::deque<size_t> todo;
    std::set<size_t> done_idxs;
    todo.push_back(center);
    //update u_env of all downstream sites
    while(!todo.empty()){ //iterate over multiset repeatedly until all idxs processed
        size_t idx=todo.front();
        todo.pop_front();
        bond key;
        key.todo()=0;
        key.order()=idx;
        key.depth()=g.vs()[idx].depth();
        key.bmi()=-1e50;
        key.virt_count()=2;
        auto it=g.es().lower_bound(key);
        //each time, update the u_env of the left and right child
        {
            if(done_idxs.find((*it).v1())==done_idxs.end()){ //skip if finished
                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                    todo.push_back((*it).v1());
                }
                for(size_t s=0;s<n_samples;s++){
                    array1d<double> res_vec((*it).w().nx());
                    for(size_t i=0;i<(*it).w().nx();i++){
                        std::vector<double> res_vec_addends;
                        for(size_t j=0;j<(*it).w().ny();j++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                                    res_vec_addends.push_back(r_env[idx][s].at(j)+u_env[idx][s].at(k)+(*it).w().at(i,j,k)); //log space
                                }
                                else{
                                    res_vec_addends.push_back(u_env[idx][s].at(k)+(*it).w().at(i,j,k));
                                }
                            }
                        }
                        res_vec.at(i)=lse(res_vec_addends); //log space
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
                for(size_t s=0;s<n_samples;s++){
                    array1d<double> res_vec((*it).w().ny());
                    for(size_t j=0;j<(*it).w().ny();j++){
                        std::vector<double> res_vec_addends;
                        for(size_t i=0;i<(*it).w().nx();i++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                                    res_vec_addends.push_back(l_env[idx][s].at(i)+u_env[idx][s].at(k)+(*it).w().at(i,j,k)); //log space
                                }
                                else{
                                    res_vec_addends.push_back(u_env[idx][s].at(k)+(*it).w().at(i,j,k));
                                }
                            }
                        }
                        res_vec.at(j)=lse(res_vec_addends); //log space
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
    key.virt_count()=2;
    auto it2=g.es().lower_bound(key);
    bond center_bond=*it2;
    size_t top_idx=g.vs().size()-1;
    if((*it2).order()!=top_idx){
        while(1){ // each time, update the l_env or r_env of the current tensor
            size_t u_idx=g.vs()[(*it2).order()].u_idx();
            for(size_t s=0;s<n_samples;s++){
                array1d<double> res_vec(g.vs()[(*it2).order()].rank());
                for(size_t k=0;k<(*it2).w().nz();k++){
                    std::vector<double> res_vec_addends;
                    for(size_t i=0;i<(*it2).w().nx();i++){
                        for(size_t j=0;j<(*it2).w().ny();j++){
                            res_vec_addends.push_back(l_env[(*it2).order()][s].at(i)+r_env[(*it2).order()][s].at(j)+(*it2).w().at(i,j,k)); //log space
                        }
                    }
                    res_vec.at(k)=lse(res_vec_addends); //log space
                }
                if(g.vs()[u_idx].l_idx()==(*it2).order()){
                    l_env[u_idx][s]=res_vec;
                }
                else{
                    r_env[u_idx][s]=res_vec;
                }
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
            key.virt_count()=2;
            it2=g.es().lower_bound(key);
        }
    }
    while(!todo.empty()){ //iterate over multiset repeatedly until all idxs processed
        size_t idx=todo.front();
        todo.pop_front();
        bond key;
        key.todo()=0;
        key.order()=idx;
        key.depth()=g.vs()[idx].depth();
        key.bmi()=-1e50;
        key.virt_count()=2;
        auto it=g.es().lower_bound(key);
        //each time, update the u_env of the left and right child
        {
            if(done_idxs.find((*it).v1())==done_idxs.end()){ //skip if finished
                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                    todo.push_back((*it).v1());
                }
                for(size_t s=0;s<n_samples;s++){
                    array1d<double> res_vec((*it).w().nx());
                    for(size_t i=0;i<(*it).w().nx();i++){
                        std::vector<double> res_vec_addends;
                        for(size_t j=0;j<(*it).w().ny();j++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                if(g.vs()[(*it).v1()].depth()!=0){ //not input tensor
                                    res_vec_addends.push_back(r_env[idx][s].at(j)+u_env[idx][s].at(k)+(*it).w().at(i,j,k)); //log space
                                }
                                else{
                                    res_vec_addends.push_back(u_env[idx][s].at(k)+(*it).w().at(i,j,k)); //log space
                                }
                            }
                        }
                        res_vec.at(i)=lse(res_vec_addends); //log space
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
                for(size_t s=0;s<n_samples;s++){
                    array1d<double> res_vec((*it).w().ny());
                    for(size_t j=0;j<(*it).w().ny();j++){
                        std::vector<double> res_vec_addends;
                        for(size_t i=0;i<(*it).w().nx();i++){
                            for(size_t k=0;k<(*it).w().nz();k++){
                                if(g.vs()[(*it).v2()].depth()!=0){ //not input tensor
                                    res_vec_addends.push_back(l_env[idx][s].at(i)+u_env[idx][s].at(k)+(*it).w().at(i,j,k)); //log space
                                }
                                else{
                                    res_vec_addends.push_back(u_env[idx][s].at(k)+(*it).w().at(i,j,k)); //log space
                                }
                            }
                        }
                        res_vec.at(j)=lse(res_vec_addends); //log space
                    }
                    u_env[(*it).v2()][s]=res_vec;
                }
                done_idxs.insert((*it).v2());
            }
        }
    }
    
    std::vector<double> w(n_samples);
    auto it4=g.es().rbegin();
    for(size_t s=0;s<n_samples;s++){
        array1d<double> res_vec(g.vs()[(*it4).order()].rank());
        for(size_t k=0;k<(*it4).w().nz();k++){
            std::vector<double> res_vec_addends;
            for(size_t i=0;i<(*it4).w().nx();i++){
                for(size_t j=0;j<(*it4).w().ny();j++){
                    res_vec_addends.push_back(l_env[(*it4).order()][s].at(i)+r_env[(*it4).order()][s].at(j)+(*it4).w().at(i,j,k)); //log space
                }
            }
            res_vec.at(k)=lse(res_vec_addends); //log space
        }
        std::vector<double> w_addends;
        for(size_t k=0;k<g.vs()[top_idx].rank();k++){
            w_addends.push_back(res_vec.at(k));
        }
        w[s]=lse(w_addends);
    }
    return w;
}
template std::vector<double> update_cache_w(graph<bmi_comparator>&,size_t,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::vector<std::vector<array1d<double> > >&,std::set<size_t>&);