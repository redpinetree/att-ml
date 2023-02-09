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

template<typename cmp>
void algorithm::tree_approx(size_t q,graph<cmp>& g){
    stopwatch sw1,sw2,sw3;
    // std::cout<<std::string(g)<<"\n";
    //graph deformation
    size_t iteration=0;
    while((*(g.es().rbegin())).todo()){
        // sw1.start();
        bond current=*(g.es().rbegin());
        g.es().erase((++(g.es().rbegin())).base());
        size_t master,slave;
        if(g.vs()[current.v1()].vol()<g.vs()[current.v2()].vol()){
            master=current.v2();
            slave=current.v1();
        }
        else{
            master=current.v1();
            slave=current.v2();
        }
        g.vs()[master].vol()+=g.vs()[slave].vol();
        // sw1.split();
        // std::cout<<std::string(current)<<","<<master<<","<<slave<<"\n";
        // std::cout<<std::string(g)<<"\n";
        // sw2.start();
        g.vs()[master].adj().erase(current); //remove the copy of the current edge from the slave site
        g.vs()[slave].adj().erase(current); //remove the copy of the current edge from the slave site
        // std::cout<<std::string(g)<<"\n";
        
        while(!g.vs()[slave].adj().empty()){
            size_t i=0;
            // std::cout<<"\n";
            bond slave_bond=*(g.es().find(*(g.vs()[slave].adj().begin())));
            // std::cout<<g.es().key_comp().q<<"\n";
            bond slave_bond_cpy=slave_bond;
            auto loc1=g.es().find(slave_bond);
            if(loc1!=g.es().end()){
                g.es().erase(loc1); //remove the edge from the edge list
            }
            auto loc2=g.vs()[slave].adj().find(slave_bond);
            if(loc2!=g.vs()[slave].adj().end()){
                g.vs()[slave].adj().erase(loc2); //remove the edge copy from the slave site
            }
            // std::cout<<std::string(slave_bond)<<",";
            size_t other_site;
            if(slave_bond.v1()==slave){ //slave_bond always has todo=true
                other_site=slave_bond.v2();
                slave_bond.v1()=master; //reconnect
            }
            else if(slave_bond.v2()==slave){
                other_site=slave_bond.v1();
                slave_bond.v2()=master; //reconnect
            }
            if(slave_bond.v2()<slave_bond.v1()){ //ordering
                std::swap(slave_bond.v1(),slave_bond.v2());
            }
            slave_bond.j()=renorm_coupling(q,slave_bond.j(),current.j()); //update bond weight
            slave_bond.bmi(q,slave_bond.j()); //update edge
            //merge identical bonds
            if(g.vs()[master].adj().find(slave_bond)!=g.vs()[master].adj().end()){
                slave_bond.j()+=(*(g.vs()[master].adj().find(slave_bond))).j();
                auto loc3=g.vs()[master].adj().find(slave_bond);
                bond master_bond=*loc3;
                // std::cout<<"merging: "<<std::string(slave_bond_cpy)<<"+"<<std::string(master_bond)<<"->"<<std::string(slave_bond)<<"\n";
                g.vs()[master].adj().erase(loc3); //remove the edge copy from the master site
                auto loc4=g.es().find(master_bond);
                if(loc4!=g.es().end()){
                    g.es().erase(loc4); //remove the edge from the edge list
                }
                auto loc5=g.vs()[other_site].adj().find(master_bond);
                if(loc5!=g.vs()[other_site].adj().end()){
                    g.vs()[other_site].adj().erase(loc5); //remove the edge copy from the other site (rel. to slave)
                }
            }
            slave_bond.bmi(q,slave_bond.j());
            g.vs()[master].adj().insert(slave_bond);
            auto loc6=g.vs()[other_site].adj().find(slave_bond_cpy);
            if(loc6!=g.vs()[other_site].adj().end()){
                g.vs()[other_site].adj().erase(loc6); //remove the edge copy from the other site (rel. to slave)
            }
            g.vs()[other_site].adj().insert(slave_bond);
            g.es().insert(slave_bond);
        }
        current.order()=iteration;
        current.todo()=false;
        g.es().insert(current);
        // std::cout<<std::string(g)<<"\n";
        // sw2.split();
        iteration++;
        // std::cout<<std::string(g)<<"\n";
        // exit(1);
    }
    // std::cout<<"volume time: "<<sw1.elapsed()<<"\n";
    // std::cout<<"reconnect time: "<<sw2.elapsed()<<"\n";
}
template void algorithm::tree_approx(size_t q,graph<coupling_comparator>& g);
template void algorithm::tree_approx(size_t q,graph<bmi_comparator>& g);