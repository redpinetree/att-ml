#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "algorithm.hpp"
#include "bond.hpp"
#include "site.hpp"
#include "stopwatch.hpp"

double renorm_coupling(size_t q,double k1,double k2){
    double num=exp(k1+k2)+(q-1);
    double denom=exp(k1)+exp(k2)+(q-2);
    return log(num/denom);
}

//default cmd function (alg. 1)
size_t algorithm::f(site v_i,size_t s_i,site v_j,size_t s_j){
    return (v_i.vol()>=v_j.vol())?s_i:s_j;
}

//cmd function that maximizes the entropy
array2d<size_t> algorithm::f_maxent(site v_i,site v_j,array2d<double> w,size_t r_k){
    array2d<size_t> f_res(v_i.rank(),v_j.rank());
    std::vector<double> w_sums(r_k);
    // std::cout<<v_i.rank()*v_j.rank()<<"\n";
    for(size_t n=0;n<v_i.rank()*v_j.rank();n++){
        auto max_it=std::max_element(w.e().begin(),w.e().end());
        double max=*max_it;
        size_t argmax=std::distance(w.e().begin(),max_it);
        size_t i=argmax%w.nx();
        size_t j=argmax/w.nx();
        size_t k=std::distance(w_sums.begin(),std::min_element(w_sums.begin(),w_sums.end()));
        w_sums[k]+=max;
        // std::cout<<v_i.vol()<<" "<<v_j.vol()<<" "<<i<<" "<<j<<" "<<k<<" "<<w_sums[k]<<"\n";
        w.e()[argmax]=-1; //weights can never be negative so this element will never be the max. also, passed by value so no change to original.
        f_res.at(i,j)=k;
    }
    double S=0;
    for(size_t n=0;n<w_sums.size();n++){
        S-=w_sums[n]*log(w_sums[n]);
    }
    // std::cout<<"entropy S="<<S<<"\n";
    return f_res;
}

template<typename cmp>
void algorithm::cmd_approx(size_t q,graph<cmp>& g,size_t r_max){
    std::ofstream ij_ofs("pair_ij_eff_k_dump.txt");
    std::ofstream ki_ofs("pair_ki_eff_k_dump.txt");
    r_max=(r_max==0)?q:r_max;
    //graph deformation
    size_t iteration=0;
    while((*(g.es().rbegin())).todo()){
        // sw1.start();
        bond current=*(g.es().rbegin());
        g.es().erase((++(g.es().rbegin())).base());
        
        // std::cout<<"alg:\n"<<std::string(current.f())<<"\n";
        // std::cout<<"maxent:\n"<<std::string(algorithm::f_maxent(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k))<<"\n";
        //TODO: resolve difference between f_maxent() and f() if possible
        // current.f()=algorithm::f_maxent(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k);
        // std::cout<<std::string(f)<<"\n";
        
        //determine master and slave node
        size_t master,slave;
        if(g.vs()[current.v1()].vol()==g.vs()[current.v2()].vol()){
            if(current.v1_orig()>current.v2_orig()){
                master=current.v2();
                slave=current.v1();
            }
            else{
                master=current.v1();
                slave=current.v2();
            }
        }
        else{
            if(g.vs()[current.v1()].vol()<g.vs()[current.v2()].vol()){
                master=current.v2();
                slave=current.v1();
            }
            else{
                master=current.v1();
                slave=current.v2();
            }
        }
        g.vs()[master].vol()+=g.vs()[slave].vol();
        //create new virtual site
        size_t r_k=fmin(g.vs()[master].rank()*g.vs()[slave].rank(),r_max);
        g.vs().push_back(site(r_k,g.vs()[master].vol(),current.v1(),current.v2()));
        size_t virtual_idx=g.vs().size()-1;
        //update volume of relevant downstream sites associated with new virtual site for algorithm 1
        size_t current_site=master;
        size_t next_site=master;
        while(g.vs()[current_site].virt()){
            current_site=next_site;
            if(g.vs()[current_site].virt()){
                next_site=(g.vs()[g.vs()[current_site].p1()].vol()<g.vs()[g.vs()[current_site].p2()].vol())?g.vs()[current_site].p2():g.vs()[current_site].p1();
                g.vs()[next_site].vol()=g.vs()[current_site].vol();
            }
        }
        
        //determine cmd function f()
        array2d<size_t> f(current.w().nx(),current.w().ny());
        for(size_t i=0;i<f.nx();i++){
            for(size_t j=0;j<f.ny();j++){
                f.at(i,j)=algorithm::f(g.vs()[current.v1()],i,g.vs()[current.v2()],j);
            }
        }
        current.f()=f;
        
        // sw1.split();
        // std::cout<<std::string(current)<<","<<master<<","<<slave<<","<<g.vs()[master].adj().size()<<","<<g.vs()[slave].adj().size()<<"\n";
        // std::cout<<std::string(current)<<","<<r_k<<"\n";
        // std::cout<<std::string(g)<<"\n";
        // std::cout<<iteration<<"\n";
        // sw2.start();
        std::vector<bond> cluster;
        std::vector<std::pair<size_t,size_t> > prev_idxs; //for identifying shifted bonds in alg.1 simplification
        std::vector<std::pair<size_t,size_t> > dupes;
        g.vs()[master].adj().erase(current); //remove the copy of the current edge from the master site
        g.vs()[slave].adj().erase(current); //remove the copy of the current edge from the slave site
        
        //confluent mapping decomposition
        // std::cout<<"\ntemporary removal from edgelist and reconnection:\n";
        while((!g.vs()[slave].adj().empty())||(!g.vs()[master].adj().empty())){
            // for(auto it=g.vs()[slave].adj().begin();it!=g.vs()[slave].adj().end();++it){
                // std::cout<<std::string(*it)<<"\n";
            // }
            size_t i=0;
            size_t source,not_source;
            bond current_bond;
            if(!g.vs()[slave].adj().empty()){
                current_bond=*(g.vs()[slave].adj().begin());
                source=slave;
                not_source=master;
            }
            else{
                current_bond=*(g.vs()[master].adj().begin());
                source=master;
                not_source=slave;
            }
            // std::cout<<"current bond: "<<std::string(current_bond)<<"\n";
            auto it=g.es().find(*(g.vs()[source].adj().begin()));
            if(it==g.es().end()){
                // std::cout<<"could not find bond in edgelist, skipping...\n";
                g.vs()[source].adj().erase(g.vs()[source].adj().begin());
                continue;
            }
            
            bond current_bond_cpy=current_bond;
            auto loc1=g.es().find(current_bond);
            if(loc1!=g.es().end()){
                g.es().erase(loc1); //remove the edge from the edge list
            }
            auto loc2=g.vs()[source].adj().find(current_bond);
            if(loc2!=g.vs()[source].adj().end()){
                g.vs()[source].adj().erase(loc2); //remove the edge copy from the source site
            }
            
            prev_idxs.push_back(std::pair<size_t,size_t>(current_bond.v1(),current_bond.v2()));
            size_t other_site;
            if((current_bond.v1()==current.v1())||(current_bond.v1()==current.v2())){ //current_bond always has todo=true
                // current_bond.v1_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                other_site=current_bond.v2();
                current_bond.v1()=virtual_idx; //reconnect
            }
            else if((current_bond.v2()==current.v1())||(current_bond.v2()==current.v2())){
                // current_bond.v2_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                other_site=current_bond.v1();
                current_bond.v2()=virtual_idx; //reconnect
            }
            if(current_bond.v2()<current_bond.v1()){ //ordering
                std::swap(current_bond.v1(),current_bond.v2());
                std::swap(current_bond.v1_orig(),current_bond.v2_orig());
            }
            
            //all reconnected sites are connected to the new site so all dupes are in the cluster
            auto dupe_it=std::find_if(cluster.begin(),cluster.end(),[current_bond](bond b){return current_bond.v()==b.v();});
            if(dupe_it!=cluster.end()){
                size_t dupe_idx=std::distance(cluster.begin(),dupe_it);
                dupes.push_back(std::pair<size_t,size_t>(dupe_idx,cluster.size()));
            }
            
            auto loc6=g.vs()[other_site].adj().find(current_bond_cpy);
            if(loc6!=g.vs()[other_site].adj().end()){
                g.vs()[other_site].adj().erase(loc6); //remove the edge copy from the other site (rel. to not_source)
            }
            else{
                std::cout<<"loc6 FAIL\n";
            }
            cluster.push_back(current_bond);
            // g.es().insert(current_bond);
            // std::cout<<master<<" "<<slave<<" "<<virtual_idx<<" "<<source<<" "<<not_source<<" "<<other_site<<"\n";
        }
        
        //optimization of weights in spin cluster
        // std::cout<<"\noptimization of weights in spin cluster:\n";
        // std::cout<<"current: "<<std::string(current)<<"\n";
        // std::cout<<"cluster edges:\n";
        // for(size_t n=0;n<cluster.size();n++){
            // std::cout<<std::string(cluster[n])<<"\n";
        // }
        // DEBUG: just recompute w based on renormed j if conn. to slave.
        for(size_t n=0;n<cluster.size();n++){
            // array2d<double> test(current_bond.w().nx(),current_bond.w().ny());
            // std::cout<<slave<<" "<<cluster[n].v1_orig()<<" "<<cluster[n].v2_orig()<<"\n";
            // if((cluster[n].v1_orig()==slave)||(cluster[n].v2_orig()==slave)){
            if((prev_idxs[n].first==slave)||(prev_idxs[n].second==slave)){
                cluster[n].j()=renorm_coupling(q,cluster[n].j(),current.j()); //update bond weight
                // double test_j=renorm_coupling(q,cluster[n].j(),cluster[n].j()); //update bond weight
                for(size_t i=0;i<q;i++){
                    for(size_t j=0;j<q;j++){
                        cluster[n].w().at(i,j)=(i==j)?(1/(q+(q*(q-1)*exp(-cluster[n].j())))):(1/((q*exp(cluster[n].j()))+(q*(q-1))));
                        // test.at(i,j)=(i==j)?(1/(q+(q*(q-1)*exp(-test_j)))):(1/((q*exp(test_j))+(q*(q-1))));
                    }
                }
                cluster[n].j(q,cluster[n].w());
                cluster[n].bmi(q,cluster[n].w());
            }
            // else{
                // test=cluster[n].w();
            // }
        }
        // std::cout<<"current: "<<std::string(current)<<"\n";
        // std::cout<<"cluster edges:\n";
        // for(size_t n=0;n<cluster.size();n++){
            // std::cout<<std::string(cluster[n])<<"\n";
        // }
        
        //merge identical bonds
        if(dupes.size()!=0){
            // std::cout<<"\nmerging double bonds:\n";
            // for(size_t i=0;i<dupes.size();i++){
                // std::cout<<"("<<dupes[i].first<<","<<dupes[i].second<<")"<<"\n";
            // }
            // for(size_t n=0;n<cluster.size();n++){
                // std::cout<<std::string(cluster[n])<<"\n";
            // }
            std::sort(dupes.begin(),dupes.end());
            for(int i=dupes.size()-1;i>=0;i--){
                // std::cout<<"merging: "<<std::string(cluster[dupes[i].second])<<"+"<<std::string(cluster[dupes[i].first]);
                //faster to do *= and /sum computations in the same nested loops
                double sum=0;
                for(size_t x=0;x<cluster[dupes[i].first].w().nx();x++){
                    for(size_t y=0;y<cluster[dupes[i].first].w().ny();y++){
                        cluster[dupes[i].first].w().at(x,y)*=cluster[dupes[i].second].w().at(x,y);
                        sum+=cluster[dupes[i].first].w().at(x,y);
                    }
                }
                for(size_t x=0;x<cluster[dupes[i].first].w().nx();x++){
                    for(size_t y=0;y<cluster[dupes[i].first].w().ny();y++){
                        cluster[dupes[i].first].w().at(x,y)/=sum;
                    }
                }
                cluster[dupes[i].first].virt_count()=g.vs()[cluster[dupes[i].first].v1()].virt()+g.vs()[cluster[dupes[i].first].v2()].virt();
                if((cluster[dupes[i].first].v1()==current.v1())||(cluster[dupes[i].first].v1()==current.v2())){
                    cluster[dupes[i].first].v1_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                }
                else{
                    cluster[dupes[i].first].v2_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                }
                cluster[dupes[i].first].j(q,cluster[dupes[i].first].w());
                cluster[dupes[i].first].bmi(q,cluster[dupes[i].first].w());
                // std::cout<<"->"<<std::string(cluster[dupes[i].first])<<"\n";
                cluster.erase(cluster.begin()+dupes[i].second);
            }
            // for(size_t n=0;n<cluster.size();n++){
                // std::cout<<std::string(cluster[n])<<"\n";
            // }
        }
        
        //reinsert cluster edges into adjs and edgelist
        for(size_t n=0;n<cluster.size();n++){
            g.vs()[cluster[n].v1()].adj().insert(cluster[n]);
            g.vs()[cluster[n].v2()].adj().insert(cluster[n]);
            g.es().insert(cluster[n]);
        }
        
        // std::cout<<"\n";
        current.order()=g.vs().size()-1;
        current.todo()=false;
        g.es().insert(current);
        // sw2.split();
        iteration++;
        // std::cout<<std::string(g)<<"\n";
        // exit(1);
    }
    // std::cout<<"volume time: "<<sw1.elapsed()<<"\n";
    // std::cout<<"reconnect time: "<<sw2.elapsed()<<"\n";
}
template void algorithm::cmd_approx(size_t,graph<coupling_comparator>&,size_t);
template void algorithm::cmd_approx(size_t,graph<bmi_comparator>&,size_t);