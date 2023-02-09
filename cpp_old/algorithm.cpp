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

void algorithm::tree_approx_old(size_t q,graph_old& g,bool sort_by_coupling){
    if(sort_by_coupling){
        std::sort(g.es().begin(),g.es().end(),coupling_comparator());
    }
    else{
        std::sort(g.es().begin(),g.es().end(),bmi_comparator(q));
    }
    // std::cout<<std::string(g)<<"\n";
    //graph deformation
    size_t iteration=0;
    while(g.es().back().todo()){
        bond current=g.es().back();
        g.es().pop_back();
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
        // std::cout<<std::string(current)<<","<<master<<","<<slave<<"\n";
        std::sort(g.es().begin(),g.es().end(),connected_to_site_comparator(slave));
        // std::cout<<std::string(g)<<"\n";
        for(size_t i=0;i<g.es().size();i++){
            if((g.es()[i].v1()==slave)&&(g.es()[i].todo())){
                g.es()[i].v1()=master; //reconnect
                if(g.es()[i].v2()<g.es()[i].v1()){ //ordering
                    std::swap(g.es()[i].v1(),g.es()[i].v2());
                }
                g.es()[i].j()=renorm_coupling(q,g.es()[i].j(),current.j()); //update bond weight
                g.es()[i].bmi(q,g.es()[i].j());
            }
            else if((g.es()[i].v2()==slave)&&(g.es()[i].todo())){
                g.es()[i].v2()=master; //reconnect
                if(g.es()[i].v2()<g.es()[i].v1()){ //ordering
                    std::swap(g.es()[i].v1(),g.es()[i].v2());
                }
                g.es()[i].j()=renorm_coupling(q,g.es()[i].j(),current.j()); //update bond weight
                g.es()[i].bmi(q,g.es()[i].j());
            }
            else{
                break;
            }
        }
        current.order()=iteration;
        current.todo()=false;
        g.es().push_back(current);
        // std::cout<<std::string(g)<<"\n";
        //merge identical bonds
        std::vector<size_t> remove_idxs;
        std::sort(g.es().begin(),g.es().end(),connected_to_site_comparator(master));
        size_t to_check=0;
        for(size_t i=0;i<g.es().size();i++){
            if((g.es()[i].v1()==master)||(g.es()[i].v2()==master)){
                to_check++;
            }
            else{
                break;
            }
        }
        std::sort(g.es().begin(),g.es().begin()+to_check,vertices_comparator());
        for(size_t i=0;i<to_check;i++){
            for(size_t j=i+1;j<to_check;j++){
                if(g.es()[i].v()==g.es()[j].v()){
                    // std::cout<<"merging "<<j<<" to "<<i<<"\n";
                    // std::cout<<g.es()[i]<<","<<g.es()[j]<<"\n";
                    g.es()[i].j()+=g.es()[j].j();
                    g.es()[i].bmi(q,g.es()[i].j());
                    remove_idxs.push_back(j);
                    // std::cout<<g.es()[i]<<"\n";
                }
                else{ //since the list is sorted, go to next i coord when no matches appear in the i coord
                    break;
                }
            }
        }
        std::sort(remove_idxs.begin(),remove_idxs.end(),std::greater<size_t>());
        for(size_t i=0;i<remove_idxs.size();i++){
            g.es().erase(g.es().begin()+remove_idxs[i]);
        }
        if(sort_by_coupling){
            std::sort(g.es().begin(),g.es().end(),coupling_comparator());
        }
        else{
            std::sort(g.es().begin(),g.es().end(),bmi_comparator(q));
        }
        iteration++;
        // std::cout<<std::string(g)<<"\n";
        // exit(1);
    }
}