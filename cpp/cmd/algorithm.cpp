#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "algorithm.hpp"
#include "../algorithm_nll.hpp"
#include "../bond.hpp"
#include "../observables.hpp"
#include "optimize.hpp"
#include "../site.hpp"
#include "../stopwatch.hpp"
#include "../utils.hpp"

template<typename cmp>
void algorithm::approx(size_t q,graph<cmp>& g,size_t r_max,size_t iter_max,double lr,size_t restarts){
    r_max=(r_max==0)?q:r_max;
    //graph deformation
    size_t iteration=0;
    while((*(g.es().rbegin())).todo()){
        // sw1.start();
        bond current=*(g.es().rbegin());
        g.es().erase((++(g.es().rbegin())).base());
        size_t r_k=fmin(g.vs()[current.v1()].rank()*g.vs()[current.v2()].rank(),r_max);
        
        //if v1 and v2 are real sites, assign default probs and m_vecs
        if(!g.vs()[current.v1()].virt()){
            g.vs()[current.v1()].p_k()=std::vector<double>(g.vs()[current.v1()].rank(),pow(g.vs()[current.v1()].rank(),-1));
            for(size_t idx=0;idx<g.vs()[current.v1()].rank();idx++){
                std::vector<double> res(g.vs()[current.v1()].rank()-1,0);
                double prob_factor=g.vs()[current.v1()].p_k()[idx];
                for(size_t i=0;i<g.vs()[current.v1()].rank();i++){
                    std::vector<double> contrib=observables::m_vec(g,current.v1(),i,idx,0);
                    for(size_t j=0;j<contrib.size();j++){
                        res[j]+=prob_factor*contrib[j];
                    }
                }
                g.vs()[current.v1()].m_vec().push_back(res);
            }
        }
        if(!g.vs()[current.v2()].virt()){
            g.vs()[current.v2()].p_k()=std::vector<double>(g.vs()[current.v2()].rank(),pow(g.vs()[current.v2()].rank(),-1));
            for(size_t idx=0;idx<g.vs()[current.v2()].rank();idx++){
                std::vector<double> res(g.vs()[current.v2()].rank()-1,0);
                double prob_factor=g.vs()[current.v2()].p_k()[idx];
                for(size_t i=0;i<g.vs()[current.v2()].rank();i++){
                    std::vector<double> contrib=observables::m_vec(g,current.v2(),i,idx,0);
                    for(size_t j=0;j<contrib.size();j++){
                        res[j]+=prob_factor*contrib[j];
                    }
                }
                g.vs()[current.v2()].m_vec().push_back(res);
            }
        }
        
        //initialize cmd function f()
        // current.f()=optimize::f_mvec_sim(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k);
        // current.f()=optimize::f_alg1(g.vs()[current.v1()],g.vs()[current.v2()]);
        // current.f()=optimize::f_maxent(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k);
        // current.f()=optimize::f_hybrid_maxent(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k);
        current.f()=optimize::f_hybrid_mvec_sim(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k);
        
        // std::cout<<std::string(g)<<"\n";
        // std::cout<<std::string(current)<<"\n";
        
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
        size_t depth=(g.vs()[master].depth()>g.vs()[slave].depth()?g.vs()[master].depth():g.vs()[slave].depth())+1;
        //create new virtual site and update u_idx of children
        g.vs().push_back(site(r_k,g.vs()[master].vol(),depth,current.v1(),current.v2()));
        g.vs()[current.v1()].u_idx()=g.vs().size()-1;
        g.vs()[current.v2()].u_idx()=g.vs().size()-1;
        size_t virtual_idx=g.vs().size()-1;
        //update volume of relevant downstream sites associated with new virtual site for algorithm 1
        size_t current_site=master;
        size_t next_site=master;
        while(g.vs()[current_site].virt()){
            current_site=next_site;
            if(g.vs()[current_site].virt()){
                next_site=(g.vs()[g.vs()[current_site].l_idx()].vol()<g.vs()[g.vs()[current_site].r_idx()].vol())?g.vs()[current_site].r_idx():g.vs()[current_site].l_idx();
                g.vs()[next_site].vol()=g.vs()[current_site].vol();
            }
        }
        
        // sw1.split();
        // sw2.start();
        std::vector<bond> cluster;
        bond old_current=current; //for debug
        std::vector<bond> old_cluster=cluster; //for debug
        std::vector<std::pair<size_t,size_t> > dupes;
        g.vs()[master].adj().erase(current); //remove the copy of the current edge from the master site
        g.vs()[slave].adj().erase(current); //remove the copy of the current edge from the slave site
        
        //confluent mapping decomposition
        // std::cout<<"\ntemporary removal from edgelist and reconnection:\n";
        while((!g.vs()[slave].adj().empty())||(!g.vs()[master].adj().empty())){
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
            auto it=g.es().find(*(g.vs()[source].adj().begin()));
            if(it==g.es().end()){
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
            
            old_cluster.push_back(current_bond);
            size_t other_site;
            if((current_bond.v1()==current.v1())||(current_bond.v1()==current.v2())){ //current_bond always has todo=true
                other_site=current_bond.v2();
                current_bond.v1()=virtual_idx; //reconnect
            }
            else if((current_bond.v2()==current.v1())||(current_bond.v2()==current.v2())){
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
            
            auto loc3=g.vs()[other_site].adj().find(current_bond_cpy);
            if(loc3!=g.vs()[other_site].adj().end()){
                g.vs()[other_site].adj().erase(loc3); //remove the edge copy from the other site (rel. to not_source)
            }
            cluster.push_back(current_bond);
        }
        
        //optimization of weights in spin cluster
        // DEBUG: just recompute w based on renormed j if conn. to slave.
        // optimize::potts_renorm(slave,old_cluster,current,cluster);
        current.cost()=optimize::opt(master,slave,r_k,g.vs(),old_current,old_cluster,current,cluster,iter_max,lr,restarts);
        
        //merge identical bonds
        if(dupes.size()!=0){
            std::sort(dupes.begin(),dupes.end());
            for(int i=dupes.size()-1;i>=0;i--){
                // std::cout<<"merging: "<<std::string(cluster[dupes[i].second])<<"+"<<std::string(cluster[dupes[i].first]);
                // std::cout<<"\n1st: "<<std::string(cluster[dupes[i].first].w())<<"\n";
                // std::cout<<"2nd: "<<std::string(cluster[dupes[i].second].w())<<"\n";
                //check if same size
                bool same_nx=cluster[dupes[i].first].w().nx()==cluster[dupes[i].second].w().nx();
                bool same_ny=cluster[dupes[i].first].w().ny()==cluster[dupes[i].second].w().ny();
                if(!(same_nx&&same_ny)){
                    std::cout<<"warning: merging bonds with different sizes!\n";
                }
                //faster to do *= and /sum computations in the same nested loops
                std::vector<double> sum_addends;
                for(size_t x=0;x<cluster[dupes[i].first].w().nx();x++){
                    for(size_t y=0;y<cluster[dupes[i].first].w().ny();y++){
                        cluster[dupes[i].first].w().at(x,y,0)+=cluster[dupes[i].second].w().at(x,y,0); //log domain means mult is add
                        sum_addends.push_back(cluster[dupes[i].first].w().at(x,y,0));
                    }
                }
                double sum=lse(sum_addends);
                for(size_t x=0;x<cluster[dupes[i].first].w().nx();x++){
                    for(size_t y=0;y<cluster[dupes[i].first].w().ny();y++){
                        cluster[dupes[i].first].w().at(x,y,0)-=sum;
                        if(cluster[dupes[i].first].w().at(x,y,0)<log(1e-100)){ //in case the weight is negative, force it to be nonnegative!
                            cluster[dupes[i].first].w().at(x,y,0)=log(1e-100);
                        }
                    }
                }
                cluster[dupes[i].first].virt_count()=g.vs()[cluster[dupes[i].first].v1()].virt()+g.vs()[cluster[dupes[i].first].v2()].virt();
                if((cluster[dupes[i].first].v1()==current.v1())||(cluster[dupes[i].first].v1()==current.v2())){
                    cluster[dupes[i].first].v1_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                }
                else{
                    cluster[dupes[i].first].v2_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                }
                cluster[dupes[i].first].bmi(cluster[dupes[i].first].w());
                cluster.erase(cluster.begin()+dupes[i].second);
                // std::cout<<"->"<<std::string(cluster[dupes[i].first])<<"\n";
                // std::cout<<std::string(cluster[dupes[i].first].w())<<"\n";
            }
        }
        
        current.depth()=depth;
        current.order()=g.vs().size()-1;
        current.todo()=false;
        
        g.es().insert(current);
        //reinsert cluster edges into adjs and edgelist
        for(size_t n=0;n<cluster.size();n++){
            g.vs()[cluster[n].v1()].adj().insert(cluster[n]);
            g.vs()[cluster[n].v2()].adj().insert(cluster[n]);
            g.es().insert(cluster[n]);
        }
        
        algorithm::calculate_site_probs(g,current);
        
        g.vs()[current.order()].m_vec()=std::vector<std::vector<double> >();
        for(size_t idx=0;idx<r_k;idx++){
            std::vector<double> res(r_k-1,0);
            double prob_factor=g.vs()[current.order()].p_k()[idx];
            for(size_t i=0;i<r_k;i++){
                std::vector<double> contrib=observables::m_vec(g,current.order(),i,idx,0);
                for(size_t j=0;j<contrib.size();j++){
                    res[j]+=prob_factor*contrib[j];
                }
            }
            g.vs()[current.order()].m_vec().push_back(res);
        }
        
        // sw2.split();
        iteration++;
    }
    // std::cout<<"volume time: "<<sw1.elapsed()<<"\n";
    // std::cout<<"reconnect time: "<<sw2.elapsed()<<"\n";
}
template void algorithm::approx(size_t,graph<bmi_comparator>&,size_t,size_t,double,size_t);