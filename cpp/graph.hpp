/*
Copyright 2025 Katsuya O. Akamatsu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPH_
#define GRAPH_

#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <tuple>

#include "site.hpp"
#include "bond.hpp"

template<typename cmp>
class graph{
public:
    graph(){
        this->n_phys_sites_=this->vs().size();
        this->center_idx_=this->vs().size()-1;
    }
    graph(std::vector<site> vs,std::multiset<bond,cmp> es):vs_(vs),es_(es){
        this->n_phys_sites_=this->vs().size();
        this->center_idx_=this->vs().size()-1;
    };
    operator std::string() const{
        std::ostringstream vs_ss,es_ss;
        for(int i=0;i<this->vs().size();i++){
            vs_ss << std::string(this->vs()[i]);
            if(i!=this->vs().size()-1){
                vs_ss << ", ";
            }
        }
        int i=0;
        std::multiset<bond,cmp> es=this->es();
        for(auto it=es.begin();it!=es.end();++it){
            es_ss << std::string(*it);
            if(i!=es.size()-1){
                es_ss << ", ";
            }
            i++;
        }
        return "vertices: ["+vs_ss.str()+"]\nedges: ["+es_ss.str()+"]\n";
    }
    const std::vector<site>& vs() const{return this->vs_;};
    const std::multiset<bond,cmp>& es() const{return this->es_;};
    int n_phys_sites() const{return this->n_phys_sites_;};
    std::vector<site>& vs(){return this->vs_;};
    std::multiset<bond,cmp>& es(){return this->es_;};
    int& n_phys_sites(){return this->n_phys_sites_;};
    int center_idx() const{return this->center_idx_;};
    int& center_idx(){return this->center_idx_;};

    //ver. 1 2025/01/29 - initial
    void save(std::ostream& os){
        int ver=1;
        os.write(reinterpret_cast<const char*>(&ver),sizeof(ver)); //version
        os.write(reinterpret_cast<const char*>(&(this->n_phys_sites())),sizeof(this->n_phys_sites()));
        os.write(reinterpret_cast<const char*>(&(this->center_idx())),sizeof(this->center_idx()));
        //write es
        int es_size=this->es().size();
        os.write(reinterpret_cast<const char*>(&es_size),sizeof(es_size)); //version
        for(auto it=this->es().begin();it!=this->es().end();++it){
            bond b=*it;
            b.save(os);
        }
        //write vs
        int vs_size=this->vs().size();
        os.write(reinterpret_cast<const char*>(&vs_size),sizeof(vs_size)); //version
        for(auto it=this->vs().begin();it!=this->vs().end();++it){
            site s=*it;
            s.save(os);
        }
    }
    
    static graph<cmp> load(std::istream& is){
        graph<cmp> g;
        int ver,n_phys_sites,center_idx,es_size,vs_size;
        std::multiset<bond,cmp> es;
        is.read(reinterpret_cast<char*>(&ver),sizeof(ver)); //version
        if(ver!=1){
            std::cout<<"Wrong input version! Expected v1 and got v"<<ver<<".\n";
            exit(1);
        }
        is.read(reinterpret_cast<char*>(&n_phys_sites),sizeof(n_phys_sites));
        is.read(reinterpret_cast<char*>(&center_idx),sizeof(center_idx));
        g.n_phys_sites()=n_phys_sites;
        g.center_idx()=center_idx;
        is.read(reinterpret_cast<char*>(&es_size),sizeof(es_size));
        for(int i=0;i<es_size;i++){
            es.insert(bond::load(is));
        }
        g.es()=es;
        is.read(reinterpret_cast<char*>(&vs_size),sizeof(vs_size));
        std::vector<site> vs(vs_size);
        for(int i=0;i<vs_size;i++){
            vs[i]=site::load(is);
        }
        g.vs()=vs;
        return g;
    }

private:
    std::vector<site> vs_;
    std::multiset<bond,cmp> es_;
    int n_phys_sites_;
    int center_idx_; //for born machine, should be top idx in single-layer scheme
};

#endif
