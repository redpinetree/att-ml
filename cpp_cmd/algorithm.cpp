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
    r_max=(r_max==0)?q:r_max;
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
        
        // sw1.split();
        // std::cout<<std::string(current)<<","<<master<<","<<slave<<","<<g.vs()[master].adj().size()<<","<<g.vs()[slave].adj().size()<<"\n";
        // std::cout<<std::string(current)<<","<<r_k<<"\n";
        // std::cout<<std::string(g)<<"\n";
        // std::cout<<iteration<<"\n";
        // sw2.start();
        g.vs()[master].adj().erase(current); //remove the copy of the current edge from the master site
        g.vs()[slave].adj().erase(current); //remove the copy of the current edge from the slave site
        
        //determine cmd function f()
        array2d<size_t> f(current.w().nx(),current.w().ny());
        for(size_t i=0;i<f.nx();i++){
            for(size_t j=0;j<f.ny();j++){
                f.at(i,j)=algorithm::f(g.vs()[current.v1()],i,g.vs()[current.v2()],j);
            }
        }
        current.f()=f;
        // std::cout<<"alg:\n"<<std::string(current.f())<<"\n";
        // std::cout<<"maxent:\n"<<std::string(algorithm::f_maxent(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k))<<"\n";
        //TODO: resolve difference between f_maxent() and f() if possible
        // current.f()=algorithm::f_maxent(g.vs()[current.v1()],g.vs()[current.v2()],current.w(),r_k);
        // std::cout<<std::string(f)<<"\n";
        
        //precompute intermediate quantities for weight optimization
        std::vector<double> g1(current.w().nx(),1);
        std::vector<double> g2(current.w().ny(),1);
        for(size_t i=0;i<current.w().nx();i++){
            for(auto it=g.vs()[current.v1()].adj().begin();it!=g.vs()[current.v1()].adj().end();++it){
                // std::cout<<"g1_contrib: "<<((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i]<<"\n";
                g1[i]*=((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i];
            }
        }
        for(size_t j=0;j<current.w().ny();j++){
            for(auto it=g.vs()[current.v2()].adj().begin();it!=g.vs()[current.v2()].adj().end();++it){
                // std::cout<<"g2_contrib: "<<((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[i]<<"\n";
                g2[j]*=((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j];
            }
        }
        //calculate pair_{ij}
        //TODO: calculate pair'_{ij}^{env} (uses cm func), can be reused for pair'_{ki_mu}^{env}?
        //TODO: NaNs out with test256.txt
        array2d<double> pair_ij(current.w().nx(),current.w().ny());
        array2d<double> pair_ij_env(current.w().nx(),current.w().ny());
        double sum_pair_ij=0;
        double sum_pair_ij_env=0;
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                pair_ij.at(i,j)=current.w().at(i,j)*g1[i]*g2[j];
                sum_pair_ij+=pair_ij.at(i,j);
                double g1_env=1;
                double g2_env=1;
                size_t k=current.f().at(i,j); //select where output is added to
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
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                pair_ij.at(i,j)/=sum_pair_ij;
                pair_ij_env.at(i,j)/=sum_pair_ij_env;
            }
        }
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
                current_bond=*(g.es().find(*(g.vs()[slave].adj().begin())));
                source=slave;
                not_source=master;
            }
            else{
                current_bond=*(g.es().find(*(g.vs()[master].adj().begin())));
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
            /*
            //DEBUG: just recompute w based on renormed j if conn. to slave.
            array2d<double> test(current_bond.w().nx(),current_bond.w().ny());
            if(source==slave){
                // current_bond.j()=renorm_coupling(q,current_bond.j(),current.j()); //update bond weight
                double test_j=renorm_coupling(q,current_bond.j(),current.j()); //update bond weight
                for(size_t i=0;i<q;i++){
                    for(size_t j=0;j<q;j++){
                        // current_bond.w().at(i,j)=(i==j)?(1/(q+(q*(q-1)*exp(-current_bond.j())))):(1/((q*exp(current_bond.j()))+(q*(q-1))));
                        test.at(i,j)=(i==j)?(1/(q+(q*(q-1)*exp(-test_j)))):(1/((q*exp(test_j))+(q*(q-1))));
                    }
                }
            }
            else{
                test=current_bond.w();
            }
            // std::cout<<"rg:\n"<<std::string(current_bond.w())<<"\n";
            // std::cout<<"rg:\n"<<std::string(test)<<"\n";
            */
            
            //TODO: implement shewchuk summation, O(n)? time complexity on average, instead of sorting then summing, which is O(n log n)+O(n). but make sure to check times, since this scales with r_k (or q)
            //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
            //pair_ki,imu if source==current.v1(), pair_kj,jnu if source==current.v2(). wlog, use pair_ki and imu.
            //k, the new index, is always second because virtual indices > physical indices.
            for(size_t n=0;n<1;n++){
            array2d<double> pair_ki((source==current.v1())?current_bond.w().nx():current_bond.w().ny(),r_k);
            array2d<double> pair_ki_env((source==current.v1())?current_bond.w().nx():current_bond.w().ny(),r_k);
            double sum_pair_ki=0;
            std::vector<std::vector<double> > addends(pair_ki.nx()*pair_ki.ny());
            std::vector<std::vector<double> > addends_env(pair_ki_env.nx()*pair_ki_env.ny());
            for(size_t i=0;i<current_bond.w().nx();i++){
                for(size_t j=0;j<current_bond.w().ny();j++){
                    size_t k=current.f().at(i,j); //select where output is added to
                    double g1_ki=g1[i];
                    double g2_ki=g2[j];
                    if(source==current.v1()){
                        g1_ki/=(current_bond.w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i];
                    }
                    else{
                        g2_ki/=(current_bond.w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j];
                    }
                    // std::cout<<"g1*g2: "<<(g1_ki*g2_ki)<<"\n";
                    // std::cout<<std::string(current.w());
                    for(size_t imu=0;imu<pair_ki.nx();imu++){
                        // std::cout<<"pair_ki["<<imu<<","<<k<<"] partial: "<<current_bond.w().at(imu,(source==current.v1())?i:j)<<"\n";
                        // std::cout<<"pair_ki["<<imu<<","<<k<<"] contrib: "<<(current.w().at(i,j)*current_bond.w().at(imu,(source==current.v1())?i:j)*g1_ki*g2_ki)<<"\n";
                        //TODO: how to reinitialize current_bond.w() for r_max>q? current iteration appears to worsen the approximation...
                        addends[(k*pair_ki.nx())+imu].push_back(current.w().at(i,j)*current_bond.w().at(imu,(source==current.v1())?i:j)*g1_ki*g2_ki);
                        addends_env[(k*pair_ki_env.nx())+imu].push_back(pair_ij_env.at(i,j)*current.w().at(i,j)/current_bond.w().sum_over_axis(0)[k]);
                        // pair_ki.at(imu,k)+=current.w().at(i,j)*current_bond.w().at(imu,(source==current.v1())?i:j)*g1_ki*g2_ki;
                        // pair_ki_env.at(imu,k)+=pair_ij_env.at(i,j)*current.w().at(i,j)/current_bond.w().sum_over_axis(0)[k];
                        sum_pair_ki+=current.w().at(i,j)*current_bond.w().at(imu,(source==current.v1())?i:j)*g1_ki*g2_ki;
                    }
                }
            }
            for(size_t imu=0;imu<pair_ki.nx();imu++){
                for(size_t k=0;k<pair_ki.ny();k++){
                    std::sort(addends[(k*pair_ki.nx())+imu].begin(),addends[(k*pair_ki.nx())+imu].end());
                    std::sort(addends_env[(k*pair_ki.nx())+imu].begin(),addends_env[(k*pair_ki.nx())+imu].end());
                    double e=0;
                    double e_env=0;
                    for(size_t i=0;i<addends[(k*pair_ki.nx())+imu].size();i++){
                        // std::cout<<addends[(k*pair_ki.nx())+imu][i]<<" ";
                        e+=addends[(k*pair_ki.nx())+imu][i];
                    }
                    // std::cout<<"\n";
                    for(size_t i=0;i<addends_env[(k*pair_ki.nx())+imu].size();i++){
                        // std::cout<<addends_env[(k*pair_ki.nx())+imu][i]<<" ";
                        e_env+=addends_env[(k*pair_ki.nx())+imu][i];
                    }
                    // std::cout<<"\n";
                    pair_ki.at(imu,k)=e;
                    pair_ki_env.at(imu,k)=e_env;
                }
            }
            for(size_t i=0;i<pair_ki.nx();i++){
                for(size_t j=0;j<pair_ki.ny();j++){
                    pair_ki.at(i,j)/=sum_pair_ki;
                }
            }
            for(size_t i=0;i<pair_ki_env.nx();i++){
                for(size_t j=0;j<pair_ki_env.ny();j++){
                    current_bond.w().at(i,j)=pair_ki.at(i,j)/pair_ki_env.at(i,j);
                    // test.at(i,j)=pair_ki.at(i,j)/pair_ki_env.at(i,j);
                }
            }
            // std::cout<<"opt:\n"<<std::string(current_bond.w())<<"\n";
            // std::cout<<"opt:\n"<<std::string(test)<<"\n";
            // exit(1);
            }
            
            current_bond.j(q,current_bond.w()); //update edge
            current_bond.bmi(q,current_bond.w()); //update edge
            // std::cout<<"current_bond (idx update):"<<std::string(current_bond)<<"\n";
            
            size_t other_site;
            if((current_bond.v1()==current.v1())||(current_bond.v1()==current.v2())){ //current_bond always has todo=true
                current_bond.v1_orig()=(current.v1()==master)?current.v1_orig():current.v2_orig();
                other_site=current_bond.v2();
                current_bond.v1()=virtual_idx; //reconnect
            }
            else if((current_bond.v2()==current.v1())||(current_bond.v2()==current.v2())){
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
                else{
                    std::cout<<"loc4 FAIL\n";
                }
                auto loc5=g.vs()[virtual_idx].adj().find(master_bond);
                if(loc5!=g.vs()[virtual_idx].adj().end()){
                    g.vs()[virtual_idx].adj().erase(loc5); //remove the edge copy from the other_site (rel. to source)
                }
                else{
                    std::cout<<"loc5 FAIL\n";
                }
            }
            
            auto loc6=g.vs()[other_site].adj().find(current_bond_cpy);
            if(loc6!=g.vs()[other_site].adj().end()){
                g.vs()[other_site].adj().erase(loc6); //remove the edge copy from the other site (rel. to not_source)
            }
            else{
                std::cout<<"loc6 FAIL\n";
            }
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
template void algorithm::cmd_approx(size_t,graph<coupling_comparator>&,size_t);
template void algorithm::cmd_approx(size_t,graph<bmi_comparator>&,size_t);