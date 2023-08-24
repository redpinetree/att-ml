#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <string>

#include "observables.hpp"
#include "ndarray.hpp"

using namespace std::complex_literals;

//default cmd function (alg. 1)
size_t f(site v_i,size_t s_i,site v_j,size_t s_j){
    return (v_i.vol()>=v_j.vol())?s_i:s_j;
}

//potts weight
double w(size_t q,size_t i,size_t j,double k){
    return (i==j)?1/(q+(q*(q-1)*exp(-k))):1/((q*exp(k))+(q*(q-1)));
}

//cartesian product of identical vectors [0,...,q-1], p times
std::vector<std::vector<size_t> > spin_cart_prod(size_t q, size_t p){
    std::vector<std::vector<size_t> > cart_prod;
    std::vector<size_t> term(p,0);
    size_t total=1;
    for(size_t i=0;i<p;i++){
        total*=q;
    }
    for(size_t i=0;i<total;i++){
        size_t i_cpy=i;
        for(size_t j=0;j<p;j++){
            term[p-1-j]=i_cpy%q;
            i_cpy/=q;
        }
        cart_prod.push_back(term);
    }
    return cart_prod;
}

//binomial coefficient
size_t binom(size_t n,size_t k){
    std::vector<size_t> cache(k);
    cache[0]=n-k+1;
    for(size_t i=1;i<k;i++){
        cache[i]=cache[i-1]*(cache[0]+i)/(i+1);
    }
    return cache[k-1];
}

std::vector<std::string> observables::output_lines;
std::vector<std::vector<double> > observables::probs;
std::map<std::tuple<size_t,size_t,size_t,std::vector<size_t> >,double> observables::known_factors;
std::map<std::tuple<size_t,size_t,size_t,std::vector<size_t>,std::vector<double> >,std::complex<double> > observables::known_factors_complex;

template<typename cmp>
void observables::cmd_treeify(graph<cmp>& g){
    for(auto it=g.es().begin();it!=g.es().end();++it){
        //keep track of bonds associated with each virtual site
        g.vs()[(*it).order()].adj().insert(*it);
    }
    
    //calculate probs
    observables::probs=std::vector<std::vector<double> >(g.vs().size());
    for(auto it=g.es().begin();it!=g.es().end();++it){
        if(!g.vs()[(*it).v1()].virt()){
            observables::probs[(*it).v1()]=std::vector<double>(g.vs()[(*it).v1()].rank(),pow(g.vs()[(*it).v1()].rank(),-1));
        }
        if(!g.vs()[(*it).v2()].virt()){
            observables::probs[(*it).v2()]=std::vector<double>(g.vs()[(*it).v2()].rank(),pow(g.vs()[(*it).v2()].rank(),-1));
        }
        std::vector<double> p_k(g.vs()[(*it).order()].rank(),0);
        double sum=0;
        for(size_t i=0;i<g.vs()[(*it).v1()].rank();i++){
            for(size_t j=0;j<g.vs()[(*it).v2()].rank();j++){
                double k=(*it).f().at(i,j);
                double e=(*it).w().at(i,j);
                double p=observables::probs[(*it).v1()][i]*observables::probs[(*it).v2()][j]*(*it).w().at(i,j);
                sum+=p;
                p_k[k]+=p;
            }
        }
        for(size_t k=0;k<p_k.size();k++){
            p_k[k]/=sum;
        }
        observables::probs[(*it).order()]=p_k;
    }
}
template void observables::cmd_treeify<coupling_comparator>(graph<coupling_comparator>&);
template void observables::cmd_treeify<bmi_comparator>(graph<bmi_comparator>&);

//this m function is just a wrapper for the top-down version
template<typename cmp>
double observables::m(graph<cmp>& g,size_t root,size_t n,size_t p,size_t depth){
    // std::vector<size_t> c(g.vs()[root].rank(),0);
    // c[0]=p;
    // return observables::m(g,root,n,p,c,depth);
    
    double res=0;
    std::vector<std::vector<size_t> > combos=spin_cart_prod(g.vs()[root].rank(),p);
    for(size_t idx=0;idx<combos.size();idx++){
        std::vector<size_t> combo=combos[idx];
        std::vector<size_t> c(g.vs()[root].rank(),0);
        for(size_t i=0;i<combo.size();i++){
            c[combo[i]]++;
        }
        res+=observables::m(g,root,n,p,c,depth);
    }
    return res;
}
template double observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,size_t,size_t);
template double observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t);

//this m function calculates using a top-down approach. theoretically, this is sufficient to calculate observables, but due to stack size limitations, a segfault occurs with a stack overflow for large lattices.
template<typename cmp>
double observables::m(graph<cmp>& g,size_t root,size_t n,size_t p,std::vector<size_t> c,size_t depth){
    // std::cout<<depth<<"\n";
    size_t r_k=g.vs()[root].rank();
    size_t sum_c=0;
    for(size_t i=0;i<c.size();i++){
        sum_c+=c[i];
    }
    if(sum_c!=p){
        std::cout<<"c vector should have contents sum to p.\n";
        exit(1);
    }
    //compute desired quantity, if leaf
    if(!g.vs()[root].virt()){
        double res=1;
        for(size_t i=0;i<r_k;i++){
            res*=(((n%2)==1))?pow((((double)(r_k*(i==0))-1)/(r_k-1)),c[i]):1;
            // res*=pow(2*(i==0)-1,c[i]);
        }
        // res/=(double) pow(r_k,p);
        // std::cout<<res<<"\n";
        return res;
    }
    double res=0;
    //compute probs
    size_t r_i=(*g.vs()[root].adj().begin()).w().nx();
    size_t r_j=(*g.vs()[root].adj().begin()).w().ny();
    // std::cout<<r_i<<" "<<r_j<<" "<<r_k<<" "<<c.size()<<"\n";
    if(c.size()!=r_k){
        std::cout<<"c vector should have size equal to rank of root spin.\n";
        exit(1);
    }
    //conditional prob dists for i,j given k
    array3d<double> prob_ij(r_i,r_j,r_k);
    array2d<double> prob_i(r_i,r_k);
    array2d<double> prob_j(r_j,r_k);
    std::cout<<"f:\n"<<(std::string) (*g.vs()[root].adj().begin()).f()<<"\n";
    std::cout<<"w:\n"<<(std::string) (*g.vs()[root].adj().begin()).w()<<"\n";
    array2d<size_t> test_f=(*g.vs()[root].adj().begin()).f();
    // if(test_f.nx()==2&&test_f.ny()==3){
    // test_f.at(0,0)=0;
    // test_f.at(0,1)=1;
    // test_f.at(0,2)=0;
    // test_f.at(1,0)=0;
    // test_f.at(1,1)=1;
    // test_f.at(1,2)=2;
    // test_f.at(2,0)=1;
    // test_f.at(2,1)=0;
    // test_f.at(2,2)=2;
    // }
    // std::cout<<"test_f:\n"<<(std::string) test_f<<"\n";
    // array2d<double> test_w=(*g.vs()[root].adj().begin()).w();
    // test_w.at(0,0)*=0;
    // test_w.at(0,1)*=0;
    // test_w.at(1,0)*=0;
    // test_w.at(1,1)*=0;
    // std::cout<<"test_w:\n"<<(std::string) test_w<<"\n";
    for(size_t j=0;j<r_j;j++){
        for(size_t i=0;i<r_i;i++){
            size_t k=(*g.vs()[root].adj().begin()).f().at(i,j);
            // size_t k=test_f.at(i,j);
            double e=(*g.vs()[root].adj().begin()).w().at(i,j);
            // double e=test_w.at(i,j);
            // std::cout<<i<<" "<<j<<" "<<k<<" "<<e<<"\n";
            prob_ij.at(i,j,k)=e;
            prob_i.at(i,k)+=e; //compute marginals, no norm
            prob_j.at(j,k)+=e; //compute marginals, no norm
            // prob_i.at(i,k)+=r_j*e; //compute marginals
            // prob_j.at(j,k)+=r_i*e; //compute marginals
        }
    }
    
    // std::cout<<(std::string)prob_ij<<"\n";
    // std::cout<<(std::string)prob_i<<"\n";
    // std::cout<<(std::string)prob_j<<"\n";
    for(size_t k=0;k<r_k;k++){
        double sum_ij=0;
        for(size_t i=0;i<r_i;i++){
            for(size_t j=0;j<r_j;j++){
                sum_ij+=prob_ij.at(i,j,k);
            }
        }
        if(sum_ij>1e-8){
            for(size_t i=0;i<r_i;i++){
                for(size_t j=0;j<r_j;j++){
                    prob_ij.at(i,j,k)/=sum_ij;
                }
            }
        }
    }
    for(size_t k=0;k<r_k;k++){
        double sum_i=0;
        for(size_t i=0;i<r_i;i++){
            sum_i+=prob_i.at(i,k);
        }
        if(sum_i>1e-8){
            for(size_t i=0;i<r_i;i++){
                prob_i.at(i,k)/=sum_i;
            }
        }
        double sum_j=0;
        for(size_t j=0;j<r_j;j++){
            sum_j+=prob_j.at(j,k);
        }
        if(sum_j>1e-8){
            for(size_t j=0;j<r_j;j++){
                prob_j.at(j,k)/=sum_j;
            }
        }
    }
    std::cout<<(std::string)prob_ij<<"\n";
    std::cout<<(std::string)prob_i<<"\n";
    std::cout<<(std::string)prob_j<<"\n";
    std::vector<size_t> q_idxs;
    for(size_t i=0;i<c.size();i++){
        for(size_t j=0;j<c[i];j++){
            q_idxs.push_back(i);
        }
    }
    //subtree contributions
    double c_res=0;
    std::vector<std::vector<std::vector<size_t> > > spin_combos;
    spin_combos.push_back(spin_cart_prod(r_i,p));
    spin_combos.push_back(spin_cart_prod(r_j,p));
    for(size_t down=0;down<2;down++){ //every non-leaf site has 2 downstream sites
        double temp=0;
        for(size_t s=0;s<spin_combos[down].size();s++){
            std::vector<size_t> c_vals((down==0)?r_i:r_j,0);
            for(size_t i=0;i<spin_combos[down][s].size();i++){
                c_vals[spin_combos[down][s][i]]++;
            }
            // for(size_t i=0;i<c_vals.size();i++){std::cout<<c_vals[i]<<" ";}
            //memoize
            double contrib=0;
            try{
                contrib=observables::known_factors.at(std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n,p,c_vals));
            }
            catch(const std::out_of_range& oor){
                contrib=m(g,(down==0)?g.vs()[root].p1():g.vs()[root].p2(),n,p,c_vals,depth+1);
                observables::known_factors[std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n,p,c_vals)]=contrib;
            }
            double weight=1;
            for(size_t i=0;i<p;i++){
                weight*=(down==0)?prob_i.at(spin_combos[down][s][i],q_idxs[i]):prob_j.at(spin_combos[down][s][i],q_idxs[i]);
            }
            if(depth==0){std::cout<<root<<" "<<((down==0)?g.vs()[root].p1():g.vs()[root].p2())<<" "<<n<<" "<<p<<" "<<weight<<","<<contrib<<"\n";}
            temp+=weight*contrib;
            // temp+=weight*contrib*pow(g.vs()[g.vs()[root].p1()].vol()!=1?g.vs()[g.vs()[root].p1()].vol()-1:1,p);
            // temp+=weight*contrib/((double) pow((down==0)?r_j:r_i,p));
        }
        if(depth==0){std::cout<<"subtree "<<down<<":"<<temp<<"\n";}
        c_res+=temp;
    }
    // if(depth==0){std::cout<<"c_res (subtrees):"<<c_res<<"\n";}
    res+=c_res;
    for(size_t comp=1;comp<n;comp++){
        size_t c0=comp;
        size_t c1=n-c0;
        double c_res=0;
        size_t coef=binom(n,c0)*binom(n-c0,c1);
        std::array<double,2> factors;
        for(size_t s0=0;s0<spin_combos[0].size();s0++){
            for(size_t s1=0;s1<spin_combos[1].size();s1++){
                std::vector<size_t> c_vals0(r_i,0);
                std::vector<size_t> c_vals1(r_j,0);
                for(size_t i=0;i<spin_combos[0][s0].size();i++){
                    c_vals0[spin_combos[0][s0][i]]++;
                }
                for(size_t i=0;i<spin_combos[1][s1].size();i++){
                    c_vals1[spin_combos[1][s1][i]]++;
                }
                //memoize
                try{
                    factors[0]=known_factors.at(std::make_tuple(g.vs()[root].p1(),c0,p,c_vals0));
                }
                catch(const std::out_of_range& oor){
                    factors[0]=m(g,g.vs()[root].p1(),c0,p,c_vals0,depth+1);
                    known_factors[std::make_tuple(g.vs()[root].p1(),c0,p,c_vals0)]=factors[0];
                }
                try{
                    factors[1]=known_factors.at(std::make_tuple(g.vs()[root].p2(),c1,p,c_vals1));
                }
                catch(const std::out_of_range& oor){
                    factors[1]=m(g,g.vs()[root].p2(),c1,p,c_vals1,depth+1);
                    known_factors[std::make_tuple(g.vs()[root].p2(),c1,p,c_vals1)]=factors[1];
                }
                double factor_prod=factors[0]*factors[1];
                double weight=1;
                for(size_t i=0;i<p;i++){
                    weight*=prob_ij.at(spin_combos[0][s0][i],spin_combos[1][s1][i],q_idxs[i]);
                }
                // if(depth==0){std::cout<<weight<<","<<factors[0]<<","<<factors[1]<<","<<factor_prod<<"\n";}
                c_res+=coef*weight*factor_prod;
            }
        }
        if(depth==0){std::cout<<"c_res ("<<c1<<","<<c0<<") "<<coef<<": "<<c_res<<"\n";}
        res+=c_res;
    }
    // if(depth==0){std::cout<<"res :"<<n<<" "<<p<<" "<<root<<" "<<res<<"\n";}
    // res/=pow(r_k,p);
    return res;
}
template double observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,size_t,std::vector<size_t>,size_t);
template double observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<size_t>,size_t);

//this m function is just a wrapper for the bottom-up version
template<typename cmp>
double observables::m(graph<cmp>& g,size_t root,size_t n,size_t p){
    // std::vector<size_t> c(g.vs()[root].rank(),0);
    // c[2]=p;
    // return observables::m(g,n,p,c);
    
    // size_t r_i=(*g.vs()[root].adj().begin()).w().nx();
    // size_t r_j=(*g.vs()[root].adj().begin()).w().ny();
    // size_t r_k=g.vs()[root].rank();
    // std::vector<double> p_k(r_k,0);
    // for(size_t j=0;j<r_j;j++){
        // for(size_t i=0;i<r_i;i++){
            // size_t k=(*g.vs()[root].adj().begin()).f().at(i,j);
            // double e=(*g.vs()[root].adj().begin()).w().at(i,j);
            // p_k[k]+=e;
        // }
    // }
    
    // std::cout<<"weights: ";
    // for(size_t i=0;i<g.vs()[root].rank();i++){
        // std::cout<<p_k[i]<<" ";
    // }
    // std::cout<<"\n";
    
    double res=0;
    std::vector<std::vector<size_t> > combos=spin_cart_prod(g.vs()[root].rank(),p);
    for(size_t idx=0;idx<combos.size();idx++){
        std::vector<size_t> combo=combos[idx];
        std::vector<size_t> c(g.vs()[root].rank(),0);
        double prob_factor=1;
        for(size_t i=0;i<combo.size();i++){
            c[combo[i]]++;
            // prob_factor*=p_k[combo[i]];
            std::cout<<"PROBS: ";
            for(size_t j=0;j<observables::probs[root].size();j++){
                std::cout<<observables::probs[root][j]<<" ";
            }
            std::cout<<"\n";
            prob_factor*=observables::probs[root][combo[i]];
        }
        double contrib=observables::m(g,n,p,c);
        res+=contrib*prob_factor;
        if(n){std::cout<<"contrib: " <<n<<" "<<p<<" "<<contrib<<"\n";}
    }
    // res/=pow(g.vs()[root].rank(),p);
    return res;
}
template double observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,size_t);
template double observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t);

//this m function calculates using a bottom-up approach. this fixes the stack size issue since all top-down calls resolve with depth 1. this function still requires the top-down version to compute observables as the tree is traversed towards the root.
template<typename cmp>
double observables::m(graph<cmp>& g,size_t n,size_t p,std::vector<size_t> c){
    //due to the ordering defined by the comparator, the leaves are at the start and the root is at the end of the multiset.
    for(auto it=g.es().begin();it!=g.es().end();++it){
        std::array<size_t,2> v_idxs{(*it).v1(),(*it).v2()};
        std::vector<size_t> c0(g.vs()[v_idxs[0]].rank(),0);
        c0[0]=p;
        std::vector<size_t> c1(g.vs()[v_idxs[1]].rank(),0);
        c1[0]=p;
        
        known_factors[std::make_tuple(v_idxs[0],n,p,c0)]=m(g,v_idxs[0],n,p,c0,0);
        known_factors[std::make_tuple(v_idxs[1],n,p,c1)]=m(g,v_idxs[1],n,p,c1,0);
    }
    return m(g,g.vs().size()-1,n,p,c,0);
}
template double observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,std::vector<size_t>);
template double observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,std::vector<size_t>);

//this m function is just a wrapper for the top-down version
template<typename cmp>
std::complex<double> observables::m(graph<cmp>& g,size_t root,size_t n,size_t p,std::vector<double> k,size_t depth){
    std::vector<size_t> c(g.vs()[root].rank(),0);
    c[0]=p;
    return observables::m(g,root,n,p,c,k,depth);
}
template std::complex<double> observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,size_t,std::vector<double>,size_t);
template std::complex<double> observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<double>,size_t);

//this m function calculates using a top-down approach. theoretically, this is sufficient to calculate observables, but due to stack size limitations, a segfault occurs with a stack overflow for large lattices.
template<typename cmp>
std::complex<double> observables::m(graph<cmp>& g,size_t root,size_t n,size_t p,std::vector<size_t> c,std::vector<double> k,size_t depth){
    // std::cout<<depth<<"\n";
    size_t r_k=g.vs()[root].rank();
    size_t sum_c=0;
    for(size_t i=0;i<c.size();i++){
        sum_c+=c[i];
    }
    if(sum_c!=p){
        std::cout<<"c vector should have contents sum to p.\n";
        exit(1);
    }
    //compute desired quantity, if leaf
    if(!g.vs()[root].virt()){
        std::complex<double> res=1;
        for(size_t i=0;i<r_k;i++){
            res*=(((n%2)==1))?pow((((double)(r_k*(i==0))-1)/(r_k-1)),c[i]):1;
        }
        //compute ft
        double dot=0;
        for(size_t i=0;i<k.size();i++){
            dot+=k[i]*g.vs()[root].coords()[i];
        }
        res*=(n%2==1)?exp(1i*dot):1;
        // std::cout<<n<<","<<dot<<","<<res<<"\n";
        return res;
    }
    std::complex<double> res=0;
    //compute probs
    size_t r_i=(*g.vs()[root].adj().begin()).w().nx();
    size_t r_j=(*g.vs()[root].adj().begin()).w().ny();
    // std::cout<<r_i<<" "<<r_j<<" "<<r_k<<" "<<c.size()<<"\n";
    if(c.size()!=r_k){
        std::cout<<"c vector should have size equal to rank of root spin.\n";
        exit(1);
    }
    //conditional prob dists for i,j given k
    array3d<double> prob_ij(r_i,r_j,r_k);
    array2d<double> prob_i(r_i,r_k);
    array2d<double> prob_j(r_j,r_k);
    // std::cout<<"f:\n"<<(std::string) (*g.vs()[root].adj().begin()).f()<<"\n";
    // std::cout<<"w:\n"<<(std::string) (*g.vs()[root].adj().begin()).w()<<"\n";
    for(size_t j=0;j<r_j;j++){
        for(size_t i=0;i<r_i;i++){
            size_t k=(*g.vs()[root].adj().begin()).f().at(i,j);
            // std::cout<<r_k<<" "<<(*g.vs()[root].adj().begin()).w().at(i,j)<<"\n";
            double e=(*g.vs()[root].adj().begin()).w().at(i,j);
            prob_ij.at(i,j,k)=e;
            prob_i.at(i,k)+=r_j*e; //compute marginals
            prob_j.at(j,k)+=r_i*e; //compute marginals
        }
    }
    for(size_t k=0;k<r_k;k++){
        double sum_ij=0;
        for(size_t i=0;i<r_i;i++){
            for(size_t j=0;j<r_j;j++){
                sum_ij+=prob_ij.at(i,j,k);
            }
        }
        if(sum_ij>1e-8){
            for(size_t i=0;i<r_i;i++){
                for(size_t j=0;j<r_j;j++){
                    prob_ij.at(i,j,k)/=sum_ij;
                }
            }
        }
    }
    for(size_t k=0;k<r_k;k++){
        double sum_i=0;
        for(size_t i=0;i<r_i;i++){
            sum_i+=prob_i.at(i,k);
        }
        if(sum_i>1e-8){
            for(size_t i=0;i<r_i;i++){
                prob_i.at(i,k)/=sum_i;
            }
        }
        double sum_j=0;
        for(size_t j=0;j<r_j;j++){
            sum_j+=prob_j.at(j,k);
        }
        if(sum_j>1e-8){
            for(size_t j=0;j<r_j;j++){
                prob_j.at(j,k)/=sum_j;
            }
        }
    }
    // std::cout<<(std::string)prob_ij<<"\n";
    // std::cout<<(std::string)prob_i<<"\n";
    // std::cout<<(std::string)prob_j<<"\n";
    std::vector<size_t> q_idxs;
    for(size_t i=0;i<c.size();i++){
        for(size_t j=0;j<c[i];j++){
            q_idxs.push_back(i);
        }
    }
    //subtree contributions
    std::complex<double> c_res=0;
    // std::vector<std::vector<size_t> > spin_combos=spin_cart_prod(r_k,p);
    std::vector<std::vector<std::vector<size_t> > > spin_combos;
    spin_combos.push_back(spin_cart_prod(r_i,p));
    spin_combos.push_back(spin_cart_prod(r_j,p));
    for(size_t down=0;down<2;down++){ //every non-leaf site has 2 downstream sites
        std::complex<double> temp=0;
        for(size_t s=0;s<spin_combos[down].size();s++){
            std::vector<size_t> c_vals((down==0)?r_i:r_j,0);
            for(size_t i=0;i<spin_combos[down][s].size();i++){
                c_vals[spin_combos[down][s][i]]++;
            }
            //memoize
            std::complex<double> contrib=0;
            try{
                contrib=observables::known_factors_complex.at(std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n,p,c_vals,k));
            }
            catch(const std::out_of_range& oor){
                contrib=m(g,(down==0)?g.vs()[root].p1():g.vs()[root].p2(),n,p,c_vals,k,depth+1);
                observables::known_factors_complex[std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n,p,c_vals,k)]=contrib;
            }
            double weight=1;
            for(size_t i=0;i<p;i++){
                weight*=(down==0)?prob_i.at(spin_combos[down][s][i],q_idxs[i]):prob_j.at(spin_combos[down][s][i],q_idxs[i]);
            }
            // if(depth==0){std::cout<<weight<<","<<contrib<<"\n";}
            // temp+=weight*contrib;
            temp+=weight*contrib/((double) pow((down==0)?r_j:r_i,p));
        }
        // if(depth==0){std::cout<<"subtree "<<down<<":"<<temp<<"\n";}
        c_res+=temp;
    }
    // if(depth==0){std::cout<<"c_res (subtrees):"<<c_res<<"\n";}
    res+=c_res;
    for(size_t comp=1;comp<n;comp++){
        size_t c0=comp;
        size_t c1=n-c0;
        std::complex<double> c_res=0;
        size_t coef=binom(n,c0)*binom(n-c0,c1)/2; //divide by 2 for (a,b),(b,a) with diff ft factor
        std::array<std::complex<double>,2> factors;
        for(size_t s0=0;s0<spin_combos[0].size();s0++){
            for(size_t s1=0;s1<spin_combos[1].size();s1++){
                std::vector<size_t> c_vals0(r_i,0);
                std::vector<size_t> c_vals1(r_j,0);
                for(size_t i=0;i<spin_combos[0][s0].size();i++){
                    c_vals0[spin_combos[0][s0][i]]++;
                }
                for(size_t i=0;i<spin_combos[1][s1].size();i++){
                    c_vals1[spin_combos[1][s1][i]]++;
                }
                //memoize
                try{
                    factors[0]=known_factors_complex.at(std::make_tuple(g.vs()[root].p1(),c0,p,c_vals0,k));
                }
                catch(const std::out_of_range& oor){
                    factors[0]=m(g,g.vs()[root].p1(),c0,p,c_vals0,k,depth+1);
                    known_factors_complex[std::make_tuple(g.vs()[root].p1(),c0,p,c_vals0,k)]=factors[0];
                }
                try{
                    factors[1]=known_factors_complex.at(std::make_tuple(g.vs()[root].p2(),c1,p,c_vals1,k));
                }
                catch(const std::out_of_range& oor){
                    factors[1]=m(g,g.vs()[root].p2(),c1,p,c_vals1,k,depth+1);
                    known_factors_complex[std::make_tuple(g.vs()[root].p2(),c1,p,c_vals1,k)]=factors[1];
                }
                std::complex<double> factor_prod=factors[0]*std::conj(factors[1]); //(a,b)
                factor_prod+=std::conj(factors[0])*factors[1]; //(b,a)
                double weight=1;
                for(size_t i=0;i<p;i++){
                    weight*=prob_ij.at(spin_combos[0][s0][i],spin_combos[1][s1][i],q_idxs[i]);
                }
                // if(depth==0){std::cout<<weight<<","<<factors[0]<<","<<factors[1]<<","<<factor_prod<<"\n";}
                c_res+=coef*weight*factor_prod;
            }
        }
        // if(depth==0){std::cout<<"c_res ("<<c1<<","<<c0<<") "<<coef<<": "<<c_res<<"\n";}
        res+=c_res;
    }
    // if(depth==0){std::cout<<"res :"<<res<<"\n";}
    return res;
}
template std::complex<double> observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<double>,size_t);
template std::complex<double> observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<double>,size_t);

//this m function is just a wrapper for the bottom-up version
template<typename cmp>
std::complex<double> observables::m(graph<cmp>& g,size_t root,size_t n,size_t p,std::vector<double> k){
    // std::vector<size_t> c(g.vs()[root].rank(),0);
    // c[0]=p;
    // return observables::m(g,n,p,c,k);
    
    std::complex<double> res=0;
    std::vector<std::vector<size_t> > combos=spin_cart_prod(g.vs()[root].rank(),p);
    for(size_t idx=0;idx<combos.size();idx++){
        std::vector<size_t> combo=combos[idx];
        std::vector<size_t> c(g.vs()[root].rank(),0);
        for(size_t i=0;i<combo.size();i++){
            c[combo[i]]++;
        }
        res+=observables::m(g,n,p,c,k);
    }
    return res;
}
template std::complex<double> observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,size_t,std::vector<double>);
template std::complex<double> observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<double>);

//this m function calculates using a bottom-up approach. this fixes the stack size issue since all top-down calls resolve with depth 1. this function still requires the top-down version to compute observables as the tree is traversed towards the root.
template<typename cmp>
std::complex<double> observables::m(graph<cmp>& g,size_t n,size_t p,std::vector<size_t> c,std::vector<double> k){
    //due to the ordering defined by the comparator, the leaves are at the start and the root is at the end of the multiset.
    for(auto it=g.es().begin();it!=g.es().end();++it){
        std::array<size_t,2> v_idxs{(*it).v1(),(*it).v2()};
        std::vector<size_t> c0(g.vs()[v_idxs[0]].rank(),0);
        c0[0]=p;
        std::vector<size_t> c1(g.vs()[v_idxs[1]].rank(),0);
        c1[0]=p;
        
        known_factors_complex[std::make_tuple(v_idxs[0],n,p,c0,k)]=m(g,v_idxs[0],n,p,c0,k,0);
        known_factors_complex[std::make_tuple(v_idxs[1],n,p,c1,k)]=m(g,v_idxs[1],n,p,c1,k,0);
    }
    return m(g,g.vs().size()-1,n,p,c,k,0);
}
template std::complex<double> observables::m<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t,std::vector<size_t>,std::vector<double>);
template std::complex<double> observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,std::vector<size_t>,std::vector<double>);

template<typename cmp>
void observables::print_moments(graph<cmp>& g,size_t q,size_t n_phys_sites){ //debug
    double m1_1=observables::m(g,g.vs().size()-1,1,1);
    double m1_2=observables::m(g,g.vs().size()-1,1,2);
    double m1_3=observables::m(g,g.vs().size()-1,1,3);
    double m2_1=observables::m(g,g.vs().size()-1,2,1);
    double m2_2=observables::m(g,g.vs().size()-1,2,2);
    double m2_3=observables::m(g,g.vs().size()-1,2,3);
    double m3_1=observables::m(g,g.vs().size()-1,3,1);
    double m3_2=observables::m(g,g.vs().size()-1,3,2);
    double m3_3=observables::m(g,g.vs().size()-1,3,3);
    double m4_1=observables::m(g,g.vs().size()-1,4,1);
    double m4_2=observables::m(g,g.vs().size()-1,4,2);
    double m4_3=observables::m(g,g.vs().size()-1,4,3);
    std::cout<<"sum <si>: "<<m1_1<<"/"<<n_phys_sites<<" = "<<(m1_1/n_phys_sites)<<"\n";
    std::cout<<"sum <si>^2: "<<m1_2<<"/"<<n_phys_sites<<" = "<<(m1_2/n_phys_sites)<<"\n";
    std::cout<<"sum <si>^3: "<<m1_3<<"/"<<n_phys_sites<<" = "<<(m1_3/n_phys_sites)<<"\n";
    std::cout<<"sum <si sj>: "<<m2_1<<"/"<<pow(n_phys_sites,2)<<" = "<<(m2_1/pow(n_phys_sites,2))<<"\n";
    std::cout<<"sum <si sj>^2: "<<m2_2<<"/"<<pow(n_phys_sites,2)<<" = "<<(m2_2/pow(n_phys_sites,2))<<"\n";
    std::cout<<"sum <si sj>^3: "<<m2_3<<"/"<<pow(n_phys_sites,2)<<" = "<<(m2_3/pow(n_phys_sites,2))<<"\n";
    std::cout<<"sum <si sj sk>: "<<m3_1<<"/"<<pow(n_phys_sites,3)<<" = "<<(m3_1/pow(n_phys_sites,3))<<"\n";
    std::cout<<"sum <si sj sk>^2: "<<m3_2<<"/"<<pow(n_phys_sites,3)<<" = "<<(m3_2/pow(n_phys_sites,3))<<"\n";
    std::cout<<"sum <si sj sk>^3: "<<m3_3<<"/"<<pow(n_phys_sites,3)<<" = "<<(m3_3/pow(n_phys_sites,3))<<"\n";
    std::cout<<"sum <si sj sk sl>: "<<m4_1<<"/"<<pow(n_phys_sites,4)<<" = "<<(m4_1/pow(n_phys_sites,4))<<"\n";
    std::cout<<"sum <si sj sk sl>^2: "<<m4_2<<"/"<<pow(n_phys_sites,4)<<" = "<<(m4_2/pow(n_phys_sites,4))<<"\n";
    std::cout<<"sum <si sj sk sl>^3: "<<m4_3<<"/"<<pow(n_phys_sites,4)<<" = "<<(m4_3/pow(n_phys_sites,4))<<"\n";
}
template void observables::print_moments<coupling_comparator>(graph<coupling_comparator>&,size_t,size_t);
template void observables::print_moments<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t);

void observables::write_output(std::string fn,std::vector<std::string>& lines){
    std::ofstream ofs(fn);
    for(size_t i=0;i<lines.size();i++){
        ofs<<lines[i];
    }
}

void observables::write_output(std::vector<std::string>& lines){
    for(size_t i=0;i<lines.size();i++){
        std::cout<<lines[i];
    }
}