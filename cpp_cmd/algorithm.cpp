#include <cmath>
#include <iostream>
#include <algorithm>

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

template<typename cmp>
void algorithm::cmd_approx(size_t q,graph<cmp>& g){
    //graph deformation
    size_t iteration=0;
    while((*(g.es().rbegin())).todo()){
        // sw1.start();
        bond current=*(g.es().rbegin());
        g.es().erase((++(g.es().rbegin())).base());
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
        g.vs().push_back(site(g.vs()[master].vol(),current.v1(),current.v2()));
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
        
        // sw1.split();
        // std::cout<<std::string(current)<<","<<master<<","<<slave<<"\n";
        // std::cout<<std::string(g)<<"\n";
        // std::cout<<iteration<<"\n";
        // sw2.start();
        g.vs()[master].adj().erase(current); //remove the copy of the current edge from the master site
        g.vs()[slave].adj().erase(current); //remove the copy of the current edge from the slave site
        
        //calculate pair_{ij}
        //TODO: calculate pair'_{ij}^{env} (uses cm func), can be reused for pair'_{ki_mu}^{env}?
        array2d<double> pair_ij(current.w().nx(),current.w().ny());
        array2d<double> pair_ij_env(current.w().nx(),current.w().ny());
        double sum_pair_ij=0;
        double sum_pair_ij_env=0;
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                double g1=1;
                double g2=1;
                for(auto it=g.vs()[current.v1()].adj().begin();it!=g.vs()[current.v1()].adj().end();++it){
                    g1*=((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i];
                }
                for(auto it=g.vs()[current.v2()].adj().begin();it!=g.vs()[current.v2()].adj().end();++it){
                    g2*=((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j];
                }
                pair_ij.at(i,j)=current.w().at(i,j)*g1*g2;
                sum_pair_ij+=pair_ij.at(i,j);
                double g1_env=1;
                double g2_env=1;
                size_t k=algorithm::f(g.vs()[current.v1()],i,g.vs()[current.v2()],j); //select where output is added to
                for(auto it=g.vs()[current.v1()].adj().begin();it!=g.vs()[current.v1()].adj().end();++it){
                    g1_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                }
                for(auto it=g.vs()[current.v2()].adj().begin();it!=g.vs()[current.v2()].adj().end();++it){
                    g2_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                }
                pair_ij_env.at(i,j)=g1_env*g2_env;
                sum_pair_ij_env+=current.w().at(i,j)*pair_ij_env.at(i,j);
            }
        }
        //symmetrize
        array2d<double> sym_pair_ij(pair_ij.nx(),pair_ij.ny());
        array2d<double> sym_pair_ij_env(pair_ij_env.nx(),pair_ij_env.ny());
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                sym_pair_ij.at(i,j)=(pair_ij.at(i,j)+pair_ij.at(j,i))/(2*sum_pair_ij); //sum of elements doesnt change when symmetrizing
                sym_pair_ij_env.at(i,j)=(pair_ij_env.at(i,j)+pair_ij_env.at(j,i))/(2*sum_pair_ij_env); //sum of elements doesnt change when symmetrizing
            }
        }
        pair_ij=sym_pair_ij;
        pair_ij_env=sym_pair_ij_env;
        for(size_t i=0;i<pair_ij_env.nx();i++){
            for(size_t j=0;j<pair_ij_env.ny();j++){
                current.w().at(i,j)=pair_ij.at(i,j)/pair_ij_env.at(i,j);
            }
        }
        
        //confluent mapping decomposition
        while((!g.vs()[slave].adj().empty())||(!g.vs()[master].adj().empty())){
            size_t i=0;
            size_t source,not_source;
            bond current_bond;
            if(!g.vs()[slave].adj().empty()){
                auto it=g.es().find(*(g.vs()[slave].adj().begin()));
                if(it==g.es().end()){
                    g.vs()[slave].adj().erase(g.vs()[slave].adj().begin());
                    continue;
                }
                current_bond=*(g.es().find(*(g.vs()[slave].adj().begin())));
                source=slave;
                not_source=master;
            }
            else{
                auto it=g.es().find(*(g.vs()[master].adj().begin()));
                if(it==g.es().end()){
                    g.vs()[master].adj().erase(g.vs()[master].adj().begin());
                    continue;
                }
                current_bond=*(g.es().find(*(g.vs()[master].adj().begin())));
                source=master;
                not_source=slave;
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
            
            //DEBUG: just recompute w based on renormed j if conn. to slave.
            // if(source==slave){
                // current_bond.j()=renorm_coupling(q,current_bond.j(),current.j()); //update bond weight
                // for(size_t i=0;i<q;i++){
                    // for(size_t j=0;j<q;j++){
                        // current_bond.w().at(i,j)=(i==j)?(1/(q+(q*(q-1)*exp(-current_bond.j())))):(1/((q*exp(current_bond.j()))+(q*(q-1))));
                    // }
                // }
            // }
            
            //pair_ki,imu if source==current.v1(), pair_kj,jnu if source==current.v2(). wlog, use pair_ki and imu.
            array2d<double> pair_ki(current_bond.w().nx(),current_bond.w().ny());
            array2d<double> pair_ki_env(current_bond.w().nx(),current_bond.w().ny());
            double sum_pair_ki=0;
            for(size_t imu=0;imu<pair_ki.nx();imu++){
                for(size_t i=0;i<current_bond.w().nx();i++){
                    for(size_t j=0;j<current_bond.w().ny();j++){
                        double g1=1;
                        double g2=1;
                        size_t k=algorithm::f(g.vs()[current.v1()],i,g.vs()[current.v2()],j); //select where output is added to
                        for(auto it=g.vs()[current.v1()].adj().begin();it!=g.vs()[current.v1()].adj().end();++it){
                            if((source==current.v1())&&((*it).v()==current_bond.v())){
                                continue;
                            }
                            g1*=((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i];
                        }
                        for(auto it=g.vs()[current.v2()].adj().begin();it!=g.vs()[current.v2()].adj().end();++it){
                            if((source==current.v2())&&((*it).v()==current_bond.v())){
                                continue;
                            }
                            g2*=((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j];
                        }
                        pair_ki.at(imu,k)+=current.w().at(i,j)*current_bond.w().at(imu,(source==current.v1())?i:j)*g1*g2;
                        pair_ki_env.at(imu,k)+=current.w().at(i,j)*pair_ij_env.at(i,j)/current_bond.w().sum_over_axis(0)[k];
                        sum_pair_ki+=current.w().at(i,j)*current_bond.w().at(imu,(source==current.v1())?i:j)*g1*g2;
                    }
                }
            }
            //symmetrize
            array2d<double> sym_pair_ki(pair_ki.nx(),pair_ki.ny());
            array2d<double> sym_pair_ki_env(pair_ki_env.nx(),pair_ki_env.ny());
            for(size_t i=0;i<pair_ki.nx();i++){
                for(size_t j=0;j<pair_ki.ny();j++){
                    sym_pair_ki.at(i,j)=(pair_ki.at(i,j)+pair_ki.at(j,i))/(2*sum_pair_ki); //sum of elements doesnt change when symmetrizing
                    sym_pair_ki_env.at(i,j)=(pair_ki_env.at(i,j)+pair_ki_env.at(j,i))/2;
                }
            }
            pair_ki=sym_pair_ki;
            pair_ki_env=sym_pair_ki_env;
            for(size_t i=0;i<pair_ki_env.nx();i++){
                for(size_t j=0;j<pair_ki_env.ny();j++){
                    current_bond.w().at(i,j)=pair_ki.at(i,j)/pair_ki_env.at(i,j);
                }
            }
            
            current_bond.j(q,current_bond.w()); //update edge
            current_bond.bmi(q,current_bond.w()); //update edge
            // std::cout<<"current_bond (idx update):"<<std::string(current_bond)<<"\n";
            
            size_t other_site;
            if((current_bond.v1()==source)||(current_bond.v1()==not_source)){ //current_bond always has todo=true
                current_bond.v1_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                other_site=current_bond.v2();
                current_bond.v1()=virtual_idx; //reconnect
            }
            else if((current_bond.v2()==source)||(current_bond.v2()==not_source)){
                current_bond.v2_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                other_site=current_bond.v1();
                current_bond.v2()=virtual_idx; //reconnect
            }
            if(current_bond.v2()<current_bond.v1()){ //ordering
                std::swap(current_bond.v1(),current_bond.v2());
                std::swap(current_bond.v1_orig(),current_bond.v2_orig());
            }
            
            //merge identical bonds
            //NOTE: right now, using source/not_source. but technically, after going through the slave bonds, none of the master bonds will ever dupe...
            if(g.vs()[other_site].adj().find(current_bond)!=g.vs()[other_site].adj().end()){
                //faster to do *= and /sum computations in the same nested loops
                double sum=0;
                for(size_t x=0;x<current_bond.w().nx();x++){
                    for(size_t y=0;y<current_bond.w().ny();y++){
                        current_bond.w().at(x,y)*=(*(g.vs()[other_site].adj().find(current_bond))).w().at(x,y);
                        sum+=current_bond.w().at(x,y);
                    }
                }
                for(size_t x=0;x<current_bond.w().nx();x++){
                    for(size_t y=0;y<current_bond.w().ny();y++){
                        current_bond.w().at(x,y)/=sum;
                    }
                }
                
                current_bond.virt_count()=g.vs()[current_bond.v1()].virt()+g.vs()[current_bond.v2()].virt();
                current_bond.j(q,current_bond.w());
                current_bond.bmi(q,current_bond.w());
                auto loc3=g.vs()[other_site].adj().find(current_bond);
                bond master_bond=*loc3;
                // std::cout<<"merging: "<<std::string(current_bond_cpy)<<"+"<<std::string(master_bond)<<"->"<<std::string(current_bond)<<"\n";
                g.vs()[other_site].adj().erase(loc3); //remove the edge copy from the other_site site
                auto loc4=g.es().find(master_bond);
                if(loc4!=g.es().end()){
                    g.es().erase(loc4); //remove the edge from the edge list
                }
                // else{
                    // std::cout<<"loc4 FAIL\n";
                // }
                auto loc5=g.vs()[virtual_idx].adj().find(master_bond);
                if(loc5!=g.vs()[virtual_idx].adj().end()){
                    g.vs()[virtual_idx].adj().erase(loc5); //remove the edge copy from the other_site (rel. to source)
                }
                // else{
                    // std::cout<<"loc5 FAIL\n";
                // }
            }
            auto loc6=g.vs()[other_site].adj().find(current_bond_cpy);
            if(loc6!=g.vs()[other_site].adj().end()){
                g.vs()[other_site].adj().erase(loc6); //remove the edge copy from the other site (rel. to not_source)
            }
            // else{
                // std::cout<<"loc6 FAIL\n";
            // }
            g.vs()[other_site].adj().insert(current_bond);
            g.vs()[virtual_idx].adj().insert(current_bond);
            g.es().insert(current_bond);
            // std::cout<<master<<" "<<slave<<" "<<virtual_idx<<" "<<source<<" "<<not_source<<" "<<other_site<<"\n";
        }
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
template void algorithm::cmd_approx(size_t,graph<coupling_comparator>&);
template void algorithm::cmd_approx(size_t,graph<bmi_comparator>&);