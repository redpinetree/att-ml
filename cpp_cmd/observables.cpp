#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <string>

#include "observables.hpp"
#include "ndarray.hpp"
#include "utils.hpp"

using namespace std::complex_literals;

//prefactor for each local quantity
double m_prefactor(size_t n_target,size_t r_k){
    if(n_target==1){
        return 1;
    }
    double denom_factor=0;
    for(size_t m=1;m<=n_target;m++){
        denom_factor+=binom(n_target,m)*pow(r_k,m-1)*pow(-1,n_target-m);
    }
    denom_factor+=(n_target%2==0)?1:-1;
    if(denom_factor==0){
        return 1;
    }
    double res=pow(pow(r_k-1,n_target)/(r_k*denom_factor),1/(double)n_target);
    return res;
}

double q_prefactor(size_t n_target,size_t r_k){
    if(n_target==1){
        return 1;
    }
    double denom_factor=0;
    for(size_t m=1;m<=n_target;m++){
        denom_factor+=binom(n_target,m)*pow(r_k,m-1)*pow(-1,n_target-m);
    }
    denom_factor+=(n_target%2==0)?1:-1;
    if(denom_factor==0){
        return 1;
    }
    double res=pow(r_k,n_target)/denom_factor;
    // std::cout<<"prefactor "<<pow(r_k,n_target)<<" "<<denom_factor<<"\n";
    return res;
}

std::vector<std::string> observables::output_lines;
// std::vector<std::vector<double> > observables::probs;
std::map<std::tuple<size_t,size_t,size_t>,std::vector<double> > observables::m_vec_cache;
std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t> >,double> observables::m_known_factors;
std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double> >,std::complex<double> > observables::m_known_factors_complex;
std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t> >,double> observables::q_known_factors;
std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double> >,std::complex<double> > observables::q_known_factors_complex;

//this m function calculates using a top-down approach. theoretically, this is sufficient to calculate observables, but due to stack size limitations, a segfault occurs with a stack overflow for large lattices.
template<typename cmp>
std::vector<double> observables::m_vec(graph<cmp>& g,size_t root,size_t r,size_t c,size_t depth){
    // std::cout<<depth<<"\n";
    size_t r_k=g.vs()[root].rank();
    //compute desired quantity, if leaf
    if(!g.vs()[root].virt()){
        //determine potts basis vectors
        std::vector<std::vector<double> > v;
        for(size_t i=0;i<r_k;i++){
            v.push_back(std::vector<double>(r_k-1,0));
        }
        for(size_t i=0;i<(r_k-1);i++){
            double sum=0;
            for(size_t j=0;j<v[i].size();j++){
                sum+=v[i][j]*v[i][j];
            }
            v[i][i]=sqrt(1-sum);
            for(size_t j=i+1;j<r_k;j++){
                double dot=0;
                for(size_t k=0;k<v[i].size();k++){
                    dot+=v[i][k]*v[j][k];
                }
                v[j][i]=(-(1/(double)(r_k-1))-dot)/v[i][i];
            }
        }
        // for(size_t i=0;i<v.size();i++){
            // std::cout<<"[";
            // for(size_t j=0;j<v[i].size();j++){
                // std::cout<<v[i][j]<<" ";
            // }
            // std::cout<<"]\n";
        // }
        // std::cout<<"\n";
        
        std::vector<double> res(r_k-1,0);
        if(c==r){
            res=v[c];
        }
        return res;
    }
    std::vector<double> res(r_k-1,0);
    size_t r_i=g.vs()[root].p_bond().w().nx();
    size_t r_j=g.vs()[root].p_bond().w().ny();
    //subtree contributions
    std::vector<std::vector<std::vector<size_t> > > spin_combos;
    spin_combos.push_back(spin_cart_prod(r_i,1));
    spin_combos.push_back(spin_cart_prod(r_j,1));
    for(size_t down=0;down<2;down++){ //every non-leaf site has 2 downstream sites
        for(size_t s=0;s<spin_combos[down].size();s++){
            size_t c_val=spin_combos[down][s][0];
            //memoize
            std::vector<double> contrib(r_k-1,0);
            try{
                contrib=observables::m_vec_cache.at(std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),r,c_val));
            }
            catch(const std::out_of_range& oor){
                contrib=m_vec(g,(down==0)?g.vs()[root].p1():g.vs()[root].p2(),r,c_val,depth+1);
                observables::m_vec_cache[std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),r,c_val)]=contrib;
            }
            double weight=(down==0)?g.vs()[root].p_ik().at(spin_combos[down][s][0],c):g.vs()[root].p_jk().at(spin_combos[down][s][0],c);
            // if(depth==0){std::cout<<root<<" "<<((down==0)?g.vs()[root].p1():g.vs()[root].p2())<<" "<<n<<" "<<p<<" "<<weight<<","<<contrib<<"\n";}
            for(size_t i=0;i<contrib.size();i++){
                res[i]+=weight*contrib[i];
            }
        }
    }
    // if(depth==0){std::cout<<"res :"<<n<<" "<<p<<" "<<root<<" "<<res<<"\n";}
    // res*=m_prefactor(1,q_orig);
    return res;
}
template std::vector<double> observables::m_vec<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t);

//this m function calculates using a top-down approach. theoretically, this is sufficient to calculate observables, but due to stack size limitations, a segfault occurs with a stack overflow for large lattices.
template<typename cmp>
double observables::m(graph<cmp>& g,size_t root,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c,size_t depth){
    // std::cout<<depth<<"\n";
    size_t r_k=g.vs()[root].rank();
    if(c.size()!=p){
        std::cout<<"c vector should have p elements.\n";
        exit(1);
    }
    //compute desired quantity, if leaf
    if(!g.vs()[root].virt()){
        double res=1;
        for(size_t i=0;i<p;i++){
            if(r[i]>=r_k){return 0;} //ignore when reference rank exceeds rank of spin
            // res*=pow(m_prefactor(n_target,r_k)*((double)(r_k*(c[i]==r[i]))-1)/(double)(r_k-1),n);
            // res*=pow(m_prefactor(n_target,r_k)*((double)(r_k*(c[i]==r[i]))-1)/(double)(r_k-1),n%2);
            res*=pow(((double)(r_k*(c[i]==r[i]))-1)/(double)(r_k-1),n);
        }
        return res;
    }
    double res=0;
    size_t r_i=g.vs()[root].p_bond().w().nx();
    size_t r_j=g.vs()[root].p_bond().w().ny();
    //subtree contributions
    double c_res=0;
    std::vector<std::vector<std::vector<size_t> > > spin_combos;
    spin_combos.push_back(spin_cart_prod(r_i,p));
    spin_combos.push_back(spin_cart_prod(r_j,p));
    for(size_t down=0;down<2;down++){ //every non-leaf site has 2 downstream sites
        double temp=0;
        for(size_t s=0;s<spin_combos[down].size();s++){
            std::vector<size_t> c_vals=spin_combos[down][s];
            //memoize
            double contrib=0;
            try{
                contrib=observables::m_known_factors.at(std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals));
            }
            catch(const std::out_of_range& oor){
                contrib=m(g,(down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,depth+1);
                observables::m_known_factors[std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals)]=contrib;
            }
            double weight=1;
            for(size_t i=0;i<p;i++){
                weight*=(down==0)?g.vs()[root].p_ik().at(spin_combos[down][s][i],c[i]):g.vs()[root].p_jk().at(spin_combos[down][s][i],c[i]);
            }
            // if(depth==0){std::cout<<root<<" "<<((down==0)?g.vs()[root].p1():g.vs()[root].p2())<<" "<<n<<" "<<p<<" "<<weight<<","<<contrib<<"\n";}
            temp+=weight*contrib;
        }
        // if(depth==0){std::cout<<"subtree "<<down<<":"<<temp<<"\n";}
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
                std::vector<size_t> c_vals0=spin_combos[0][s0];
                std::vector<size_t> c_vals1=spin_combos[1][s1];
                //memoize
                try{
                    factors[0]=m_known_factors.at(std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r,c_vals0));
                }
                catch(const std::out_of_range& oor){
                    factors[0]=m(g,g.vs()[root].p1(),n_target,c0,p,r,c_vals0,depth+1);
                    m_known_factors[std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r,c_vals0)]=factors[0];
                }
                try{
                    factors[1]=m_known_factors.at(std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r,c_vals1));
                }
                catch(const std::out_of_range& oor){
                    factors[1]=m(g,g.vs()[root].p2(),n_target,c1,p,r,c_vals1,depth+1);
                    m_known_factors[std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r,c_vals1)]=factors[1];
                }
                double factor_prod=factors[0]*factors[1];
                double weight=1;
                for(size_t i=0;i<p;i++){
                    weight*=g.vs()[root].p_ijk().at(spin_combos[0][s0][i],spin_combos[1][s1][i],c[i]);
                }
                // if(depth==0){std::cout<<weight<<","<<factors[0]<<","<<factors[1]<<","<<factor_prod<<"\n";}
                c_res+=coef*weight*factor_prod;
            }
        }
        // if(depth==0){std::cout<<"c_res ("<<c1<<","<<c0<<") "<<coef<<": "<<c_res<<"\n";}
        res+=c_res;
    }
    // if(depth==0){std::cout<<"res :"<<n<<" "<<p<<" "<<root<<" "<<res<<"\n";}
    return res;
}
template double observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,size_t);

//this m function is just a wrapper for the bottom-up version
template<typename cmp>
double observables::m(graph<cmp>& g,size_t q_orig,size_t root,size_t n_target,size_t p,bool abs_flag){
    // std::vector<size_t> c(g.vs()[root].rank(),0);
    // c[0]=p;
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
    // std::cout<<"PROBS: ";
    // for(size_t j=0;j<p_k.size();j++){
        // std::cout<<p_k[j]<<" ";
    // }
    // std::cout<<"\n";
    
    size_t r_k=g.vs()[root].rank();
    double res=0;
    std::vector<std::vector<size_t> > combos=spin_cart_prod(g.vs()[root].rank(),p);
    std::vector<std::vector<size_t> > r_combos=spin_cart_prod(r_k,p);
    for(size_t idx=0;idx<combos.size();idx++){
        std::vector<size_t> c=combos[idx];
        double prob_factor=1;
        for(size_t i=0;i<c.size();i++){
            // prob_factor*=p_k[c[i]];
            prob_factor*=g.vs()[root].probs()[c[i]];
        }
        double sub_res=0;
        for(size_t i=0;i<r_combos.size();i++){
            double contrib=observables::m(g,n_target,n_target,p,r_combos[i],c);
            if(abs_flag){
                sub_res=(contrib>sub_res)?contrib:sub_res;
            }
            else{
                sub_res+=contrib;
            }
            // if(n_target==2 && p==1){
                // for(size_t j=0;j<c.size();j++){
                    // std::cout<<r_combos[i][j]<<" ";
                // }
                // std::cout<<"; ";
                // for(size_t j=0;j<c.size();j++){
                    // std::cout<<c[j]<<" ";
                // }
                // std::cout<<"\n";
                // std::cout<<"contrib: " <<n_target<<" "<<p<<" "<<contrib<<" "<<prob_factor<<"\n";
            // }
        }
        res+=sub_res*prob_factor;
    }
    res*=pow(m_prefactor(n_target,q_orig),n_target*p);
    return res;
}
template double observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,bool);

//this m function calculates using a bottom-up approach. this fixes the stack size issue since all top-down calls resolve with depth 1. this function still requires the top-down version to compute observables as the tree is traversed towards the root.
template<typename cmp>
double observables::m(graph<cmp>& g,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c){
    //due to the ordering defined by the comparator, the leaves are at the start and the root is at the end of the multiset.
    for(auto it=g.es().begin();it!=g.es().end();++it){
        std::array<size_t,2> v_idxs{(*it).v1(),(*it).v2()};
        std::vector<size_t> c0(p,0);
        std::vector<size_t> c1(p,0);
        m_known_factors[std::make_tuple(v_idxs[0],n_target,n,p,r,c0)]=m(g,v_idxs[0],n_target,n,p,r,c0,0);
        m_known_factors[std::make_tuple(v_idxs[1],n_target,n,p,r,c1)]=m(g,v_idxs[1],n_target,n,p,r,c1,0);
    }
    return m(g,g.vs().size()-1,n_target,n,p,r,c,0);
}
template double observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>);

//this q function calculates using a top-down approach. theoretically, this is sufficient to calculate observables, but due to stack size limitations, a segfault occurs with a stack overflow for large lattices.
template<typename cmp>
double observables::q(graph<cmp>& g,size_t root,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c,size_t depth){
    // std::cout<<depth<<"\n";
    size_t r_k=g.vs()[root].rank();
    if(c.size()!=p){
        std::cout<<"c vector should have p elements.\n";
        exit(1);
    }
    //compute desired quantity, if leaf
    if(!g.vs()[root].virt()){
        double res=1;
        for(size_t i=0;i<n;i++){
            // if(r[i]>=r_k){return 0;} //ignore when reference rank exceeds rank of spin
            for(size_t j=0;j<p;j++){
                res*=(c[j]==r[i])?1:0;
            }
        }
        // std::cout<<res<<"\n";
        return res;
    }
    double res=0;
    size_t r_i=g.vs()[root].p_bond().w().nx();
    size_t r_j=g.vs()[root].p_bond().w().ny();
    //subtree contributions
    double c_res=0;
    std::vector<std::vector<std::vector<size_t> > > spin_combos;
    spin_combos.push_back(spin_cart_prod(r_i,p));
    spin_combos.push_back(spin_cart_prod(r_j,p));
    for(size_t down=0;down<2;down++){ //every non-leaf site has 2 downstream sites
        double temp=0;
        for(size_t s=0;s<spin_combos[down].size();s++){
            std::vector<size_t> c_vals=spin_combos[down][s];
            //memoize
            double contrib=0;
            try{
                contrib=observables::q_known_factors.at(std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals));
            }
            catch(const std::out_of_range& oor){
                contrib=q(g,(down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,depth+1);
                observables::q_known_factors[std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals)]=contrib;
            }
            double weight=1;
            for(size_t i=0;i<p;i++){
                weight*=(down==0)?g.vs()[root].p_ik().at(spin_combos[down][s][i],c[i]):g.vs()[root].p_jk().at(spin_combos[down][s][i],c[i]);
            }
            // if(depth==0){std::cout<<root<<" "<<((down==0)?g.vs()[root].p1():g.vs()[root].p2())<<" "<<n<<" "<<p<<" "<<weight<<","<<contrib<<"\n";}
            temp+=weight*contrib;
        }
        // if(depth==0){std::cout<<"subtree "<<down<<":"<<temp<<"\n";}
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
        std::vector<size_t> r0;
        std::vector<size_t> r1;
        for(size_t i=0;i<c0;i++){
            r0.push_back(r[i]);
        }
        for(size_t j=c0;j<r.size();j++){
            r1.push_back(r[j]);
        }
        for(size_t s0=0;s0<spin_combos[0].size();s0++){
            for(size_t s1=0;s1<spin_combos[1].size();s1++){
                std::vector<size_t> c_vals0=spin_combos[0][s0];
                std::vector<size_t> c_vals1=spin_combos[1][s1];
                //memoize
                try{
                    factors[0]=q_known_factors.at(std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r0,c_vals0));
                }
                catch(const std::out_of_range& oor){
                    factors[0]=q(g,g.vs()[root].p1(),n_target,c0,p,r0,c_vals0,depth+1);
                    q_known_factors[std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r0,c_vals0)]=factors[0];
                }
                try{
                    factors[1]=q_known_factors.at(std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r1,c_vals1));
                }
                catch(const std::out_of_range& oor){
                    factors[1]=q(g,g.vs()[root].p2(),n_target,c1,p,r1,c_vals1,depth+1);
                    q_known_factors[std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r1,c_vals1)]=factors[1];
                }
                double factor_prod=factors[0]*factors[1];
                double weight=1;
                for(size_t i=0;i<p;i++){
                    weight*=g.vs()[root].p_ijk().at(spin_combos[0][s0][i],spin_combos[1][s1][i],c[i]);
                }
                // if(depth==0){std::cout<<weight<<","<<factors[0]<<","<<factors[1]<<","<<factor_prod<<"\n";}
                c_res+=coef*weight*factor_prod;
            }
        }
        // if(depth==0){std::cout<<"c_res ("<<c1<<","<<c0<<") "<<coef<<": "<<c_res<<"\n";}
        res+=c_res;
    }
    // if(depth==0){std::cout<<"res :"<<n<<" "<<p<<" "<<root<<" "<<res<<"\n";}
    return res;
}
template double observables::q<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,size_t);

//this q function is just a wrapper for the bottom-up version
template<typename cmp>
double observables::q(graph<cmp>& g,size_t q_orig,size_t root,size_t n_target,size_t p,bool abs_flag){
    // std::vector<size_t> c(g.vs()[root].rank(),0);
    // c[0]=p;
    // return observables::q(g,n,p,c);
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
    // std::cout<<"PROBS: ";
    // for(size_t j=0;j<p_k.size();j++){
        // std::cout<<p_k[j]<<" ";
    // }
    // std::cout<<"\n";
    // std::cout<<"PROBS: ";
    // for(size_t j=0;j<g.vs()[root].probs().size();j++){
        // std::cout<<g.vs()[root].probs()[j]<<" ";
    // }
    // std::cout<<"\n";
    
    size_t r_k=g.vs()[root].rank();
    double res=0;
    for(size_t n=2;n<=n_target;n++){
        std::vector<std::vector<size_t> > combos=spin_cart_prod(g.vs()[root].rank(),p);
        std::vector<std::vector<size_t> > r_combos=spin_cart_prod(r_k,n);
        double n_res=0;
        for(size_t idx=0;idx<combos.size();idx++){
            std::vector<size_t> c=combos[idx];
            double prob_factor=1;
            for(size_t i=0;i<c.size();i++){
                // prob_factor*=p_k[c[i]];
                prob_factor*=g.vs()[root].probs()[c[i]];
            }
            double sub_res=0;
            for(size_t i=0;i<r_combos.size();i++){
                double contrib=observables::q(g,n,n,p,r_combos[i],c);
                if(abs_flag){
                    sub_res=(contrib>sub_res)?contrib:sub_res;
                }
                else{
                    sub_res+=contrib;
                }
                // if(n_target==2){
                    // for(size_t j=0;j<r_combos[i].size();j++){
                        // std::cout<<r_combos[i][j]<<" ";
                    // }
                    // std::cout<<"; ";
                    // for(size_t j=0;j<c.size();j++){
                        // std::cout<<c[j]<<" ";
                    // }
                    // std::cout<<"\n";
                    // std::cout<<"contrib: " <<n<<" "<<p<<" "<<contrib<<" "<<prob_factor<<"\n";
                // }
            }
            n_res+=sub_res*prob_factor;
        }
        // std::cout<<n_target<<" "<<n<<" "<<binom(n_target,n_target-n)<<" "<<pow(2,(double) n_target-n)<<" "<<pow(g.n_phys_sites(),n_target-n)<<" "<<n_res<<"\n";
        // std::cout<<(binom(n_target,n_target-n)*pow(2,(double) n-n_target)*pow(-1,(double) n))<<"\n";
        // std::cout<<(binom(n_target,n_target-n)*pow(2,(double) n-n_target)*pow(-1,(double) n)*n_res)<<"\n";
        // std::cout<<(binom(n_target,n_target-n)*pow(2,(double) n-n_target)*pow(-1,(double) n)*n_res)<<" "<<pow(g.n_phys_sites(),n_target-n)<<"\n";
        res+=pow(-1,(double) n)*binom(n_target,n_target-n)*pow(q_orig,(double) n-n_target)*pow(g.n_phys_sites(),n_target-n)*n_res;
    }
    //normalize so that perfect correlation is 1
    // std::cout<<"\n";
    // std::cout<<pow(-1,(double) n_target-1)<<" "<<(n_target-1)<<" "<<pow(2,-(double) n_target)<<" "<<pow(g.n_phys_sites(),n_target)<<"\n";
    // std::cout<<pow(-1,(double) n_target-1)*(n_target-1)*pow(2,-(double) n_target)*pow(g.n_phys_sites(),n_target)<<"\n\n";
    res+=pow(-1,(double) n_target-1)*(n_target-1)*pow(q_orig,-(double) n_target)*pow(g.n_phys_sites(),n_target);
    res*=pow(q_prefactor(n_target,q_orig),p-1);
    // std::cout<<"final_res: "<<res<<"\n";
    return res;
}
template double observables::q<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,bool);

//this q function calculates using a bottom-up approach. this fixes the stack size issue since all top-down calls resolve with depth 1. this function still requires the top-down version to compute observables as the tree is traversed towards the root.
template<typename cmp>
double observables::q(graph<cmp>& g,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c){
    //due to the ordering defined by the comparator, the leaves are at the start and the root is at the end of the multiset.
    for(auto it=g.es().begin();it!=g.es().end();++it){
        std::array<size_t,2> v_idxs{(*it).v1(),(*it).v2()};
        std::vector<size_t> c0(p,0);
        std::vector<size_t> c1(p,0);
        q_known_factors[std::make_tuple(v_idxs[0],n_target,n,p,r,c0)]=q(g,v_idxs[0],n_target,n,p,r,c0,0);
        q_known_factors[std::make_tuple(v_idxs[1],n_target,n,p,r,c1)]=q(g,v_idxs[1],n_target,n,p,r,c1,0);
    }
    return q(g,g.vs().size()-1,n_target,n,p,r,c,0);
}
template double observables::q<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>);

//this m function calculates using a top-down approach. theoretically, this is sufficient to calculate observables, but due to stack size limitations, a segfault occurs with a stack overflow for large lattices.

template<typename cmp>
std::complex<double> observables::m(graph<cmp>& g,size_t root,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c,std::vector<double> k_components,size_t depth){
    // std::cout<<depth<<"\n";
    size_t r_k=g.vs()[root].rank();
    if(c.size()!=p){
        std::cout<<"c vector should have p elements.\n";
        exit(1);
    }
    //compute desired quantity, if leaf
    if(!g.vs()[root].virt()){
        std::complex<double> res=1;
        for(size_t i=0;i<p;i++){
            if(r[i]>=r_k){return 0;} //ignore when reference rank exceeds rank of spin
            // res*=pow(m_prefactor(n_target,r_k)*((double)(r_k*(c[i]==r[i]))-1)/(double)(r_k-1),n);
            res*=pow(((double)(r_k*(c[i]==r[i]))-1)/(double)(r_k-1),n);
        }
        //compute ft
        double dot=0;
        for(size_t i=0;i<k_components.size();i++){
            dot+=k_components[i]*g.vs()[root].coords()[i];
        }
        res*=(n%2==1)?exp(1i*dot):1;
        // std::cout<<n<<","<<dot<<","<<res<<"\n";
        return res;
    }
    std::complex<double> res=0;
    size_t r_i=g.vs()[root].p_bond().w().nx();
    size_t r_j=g.vs()[root].p_bond().w().ny();
    //subtree contributions
    std::complex<double> c_res=0;
    // std::vector<std::vector<size_t> > spin_combos=spin_cart_prod(r_k,p);
    std::vector<std::vector<std::vector<size_t> > > spin_combos;
    spin_combos.push_back(spin_cart_prod(r_i,p));
    spin_combos.push_back(spin_cart_prod(r_j,p));
    for(size_t down=0;down<2;down++){ //every non-leaf site has 2 downstream sites
        std::complex<double> temp=0;
        for(size_t s=0;s<spin_combos[down].size();s++){
            std::vector<size_t> c_vals=spin_combos[down][s];
            //memoize
            std::complex<double> contrib=0;
            try{
                contrib=observables::m_known_factors_complex.at(std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,k_components));
            }
            catch(const std::out_of_range& oor){
                contrib=m(g,(down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,k_components,depth+1);
                observables::m_known_factors_complex[std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,k_components)]=contrib;
            }
            double weight=1;
            for(size_t i=0;i<p;i++){
                weight*=(down==0)?g.vs()[root].p_ik().at(spin_combos[down][s][i],c[i]):g.vs()[root].p_jk().at(spin_combos[down][s][i],c[i]);
            }
            // if(depth==0){std::cout<<weight<<","<<contrib<<"\n";}
            temp+=weight*contrib;
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
                std::vector<size_t> c_vals0=spin_combos[0][s0];
                std::vector<size_t> c_vals1=spin_combos[1][s1];
                //memoize
                try{
                    factors[0]=m_known_factors_complex.at(std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r,c_vals0,k_components));
                }
                catch(const std::out_of_range& oor){
                    factors[0]=m(g,g.vs()[root].p1(),n_target,c0,p,r,c_vals0,k_components,depth+1);
                    m_known_factors_complex[std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r,c_vals0,k_components)]=factors[0];
                }
                try{
                    factors[1]=m_known_factors_complex.at(std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r,c_vals1,k_components));
                }
                catch(const std::out_of_range& oor){
                    factors[1]=m(g,g.vs()[root].p2(),n_target,c1,p,r,c_vals1,k_components,depth+1);
                    m_known_factors_complex[std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r,c_vals1,k_components)]=factors[1];
                }
                std::complex<double> factor_prod=factors[0]*std::conj(factors[1]); //(a,b)
                factor_prod+=std::conj(factors[0])*factors[1]; //(b,a)
                double weight=1;
                for(size_t i=0;i<p;i++){
                    weight*=g.vs()[root].p_ijk().at(spin_combos[0][s0][i],spin_combos[1][s1][i],c[i]);
                }
                // if(depth==0){std::cout<<weight<<","<<factors[0]<<","<<factors[1]<<","<<factor_prod<<"\n";}
                c_res+=coef*weight*factor_prod;
            }
        }
        // if(depth==0){std::cout<<"c_res ("<<c1<<","<<c0<<") "<<coef<<": "<<c_res<<"\n";}
        res+=c_res;
    }
    // if(depth==0){std::cout<<"res :"<<res<<"\n";}
    // res*=pow((r_k-1)/(double)r_k,p);
    return res;
}
template std::complex<double> observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,std::vector<size_t> r,std::vector<size_t>,std::vector<double>,size_t);

//this m function is just a wrapper for the bottom-up version
template<typename cmp>
std::complex<double> observables::m(graph<cmp>& g,size_t q_orig,size_t root,size_t n_target,size_t p,std::vector<double> k_components,bool abs_flag){
    // std::vector<size_t> c(g.vs()[root].rank(),0);
    // c[0]=p;
    // return observables::m(g,n,p,c,k);
    
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
    // std::cout<<"PROBS: ";
    // for(size_t j=0;j<p_k.size();j++){
        // std::cout<<p_k[j]<<" ";
    // }
    // std::cout<<"\n";
    
    size_t r_k=g.vs()[root].rank();
    std::complex<double> res=0;
    std::vector<std::vector<size_t> > combos=spin_cart_prod(g.vs()[root].rank(),p);
    std::vector<std::vector<size_t> > r_combos=spin_cart_prod(r_k,p);
    for(size_t idx=0;idx<combos.size();idx++){
        std::vector<size_t> c=combos[idx];
        double prob_factor=1;
        for(size_t i=0;i<c.size();i++){
            // prob_factor*=p_k[c[i]];
            prob_factor*=g.vs()[root].probs()[c[i]];
        }
        std::complex<double> sub_res=0;
        for(size_t i=0;i<r_combos.size();i++){
            std::complex<double> contrib=observables::m(g,n_target,n_target,p,r_combos[i],c,k_components);
            if(abs_flag){
                sub_res=(std::norm(contrib)>std::norm(sub_res))?contrib:sub_res;
            }
            else{
                sub_res+=contrib;
            }
        }
        res+=sub_res*prob_factor;
        // if(n){
            // for(size_t i=0;i<c.size();i++){
                // std::cout<<c[i]<<" ";
            // }
            // std::cout<<"\n";
            // std::cout<<"contrib: " <<n<<" "<<p<<" "<<contrib<<" "<<prob_factor<<"\n";
        // }
    }
    res*=pow(m_prefactor(n_target,q_orig),n_target*p);
    return res;
}
template std::complex<double> observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,std::vector<double>,bool);

//this m function calculates using a bottom-up approach. this fixes the stack size issue since all top-down calls resolve with depth 1. this function still requires the top-down version to compute observables as the tree is traversed towards the root.
template<typename cmp>
std::complex<double> observables::m(graph<cmp>& g,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c,std::vector<double> k_components){
    //due to the ordering defined by the comparator, the leaves are at the start and the root is at the end of the multiset.
    for(auto it=g.es().begin();it!=g.es().end();++it){
        std::array<size_t,2> v_idxs{(*it).v1(),(*it).v2()};
        std::vector<size_t> c0(p,0);
        std::vector<size_t> c1(p,0);
        m_known_factors_complex[std::make_tuple(v_idxs[0],n_target,n,p,r,c0,k_components)]=m(g,v_idxs[0],n_target,n,p,r,c0,k_components,0);
        m_known_factors_complex[std::make_tuple(v_idxs[1],n_target,n,p,r,c1,k_components)]=m(g,v_idxs[1],n_target,n,p,r,c1,k_components,0);
    }
    return m(g,g.vs().size()-1,n_target,n,p,r,c,k_components,0);
}
template std::complex<double> observables::m<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double>);


//this q function calculates using a top-down approach. theoretically, this is sufficient to calculate observables, but due to stack size limitations, a segfault occurs with a stack overflow for large lattices.
template<typename cmp>
std::complex<double> observables::q(graph<cmp>& g,size_t root,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c,std::vector<double> k_components,size_t depth){
    // std::cout<<depth<<"\n";
    size_t r_k=g.vs()[root].rank();
    if(c.size()!=p){
        std::cout<<"c vector should have p elements.\n";
        exit(1);
    }
    //compute desired quantity, if leaf
    if(!g.vs()[root].virt()){
        std::complex<double> res=1;
        for(size_t i=0;i<n;i++){
            // if(r[i]>=r_k){return 0;} //ignore when reference rank exceeds rank of spin
            for(size_t j=0;j<p;j++){
                res*=(c[j]==r[i])?1:0;
            }
        }
        //compute ft
        double dot=0;
        for(size_t i=0;i<k_components.size();i++){
            dot+=k_components[i]*g.vs()[root].coords()[i];
        }
        res*=(n%2==1)?exp(1i*dot):1;
        // std::cout<<res<<"\n";
        return res;
    }
    std::complex<double> res=0;
    size_t r_i=g.vs()[root].p_bond().w().nx();
    size_t r_j=g.vs()[root].p_bond().w().ny();
    // std::cout<<r_i<<" "<<r_j<<" "<<r_k<<" "<<c.size()<<"\n";
    //subtree contributions
    std::complex<double> c_res=0;
    std::vector<std::vector<std::vector<size_t> > > spin_combos;
    spin_combos.push_back(spin_cart_prod(r_i,p));
    spin_combos.push_back(spin_cart_prod(r_j,p));
    for(size_t down=0;down<2;down++){ //every non-leaf site has 2 downstream sites
        std::complex<double> temp=0;
        for(size_t s=0;s<spin_combos[down].size();s++){
            std::vector<size_t> c_vals=spin_combos[down][s];
            //memoize
            std::complex<double> contrib=0;
            try{
                contrib=observables::q_known_factors_complex.at(std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,k_components));
            }
            catch(const std::out_of_range& oor){
                contrib=q(g,(down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,k_components,depth+1);
                observables::q_known_factors_complex[std::make_tuple((down==0)?g.vs()[root].p1():g.vs()[root].p2(),n_target,n,p,r,c_vals,k_components)]=contrib;
            }
            double weight=1;
            for(size_t i=0;i<p;i++){
                weight*=(down==0)?g.vs()[root].p_ik().at(spin_combos[down][s][i],c[i]):g.vs()[root].p_jk().at(spin_combos[down][s][i],c[i]);
            }
            // if(depth==0){std::cout<<root<<" "<<((down==0)?g.vs()[root].p1():g.vs()[root].p2())<<" "<<n<<" "<<p<<" "<<weight<<","<<contrib<<"\n";}
            temp+=weight*contrib;
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
        std::vector<size_t> r0;
        std::vector<size_t> r1;
        for(size_t i=0;i<c0;i++){
            r0.push_back(r[i]);
        }
        for(size_t j=c0;j<r.size();j++){
            r1.push_back(r[j]);
        }
        for(size_t s0=0;s0<spin_combos[0].size();s0++){
            for(size_t s1=0;s1<spin_combos[1].size();s1++){
                std::vector<size_t> c_vals0=spin_combos[0][s0];
                std::vector<size_t> c_vals1=spin_combos[1][s1];
                //memoize
                try{
                    factors[0]=q_known_factors_complex.at(std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r0,c_vals0,k_components));
                }
                catch(const std::out_of_range& oor){
                    factors[0]=q(g,g.vs()[root].p1(),n_target,c0,p,r0,c_vals0,k_components,depth+1);
                    q_known_factors_complex[std::make_tuple(g.vs()[root].p1(),n_target,c0,p,r0,c_vals0,k_components)]=factors[0];
                }
                try{
                    factors[1]=q_known_factors_complex.at(std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r1,c_vals1,k_components));
                }
                catch(const std::out_of_range& oor){
                    factors[1]=q(g,g.vs()[root].p2(),n_target,c1,p,r1,c_vals1,k_components,depth+1);
                    q_known_factors_complex[std::make_tuple(g.vs()[root].p2(),n_target,c1,p,r1,c_vals1,k_components)]=factors[1];
                }
                std::complex<double> factor_prod=factors[0]*std::conj(factors[1]); //(a,b)
                factor_prod+=std::conj(factors[0])*factors[1]; //(b,a)
                double weight=1;
                for(size_t i=0;i<p;i++){
                    weight*=g.vs()[root].p_ijk().at(spin_combos[0][s0][i],spin_combos[1][s1][i],c[i]);
                }
                // if(depth==0){std::cout<<weight<<","<<factors[0]<<","<<factors[1]<<","<<factor_prod<<"\n";}
                c_res+=coef*weight*factor_prod;
            }
        }
        // if(depth==0){std::cout<<"c_res ("<<c1<<","<<c0<<") "<<coef<<": "<<c_res<<"\n";}
        res+=c_res;
    }
    // if(depth==0){std::cout<<"res :"<<n<<" "<<p<<" "<<root<<" "<<res<<"\n";}
    return res;
}
template std::complex<double> observables::q<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double>,size_t);

//this q function is just a wrapper for the bottom-up version
template<typename cmp>
std::complex<double> observables::q(graph<cmp>& g,size_t q_orig,size_t root,size_t n_target,size_t p,std::vector<double> k_components,bool abs_flag){
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
    
    size_t r_k=g.vs()[root].rank();
    std::complex<double> res=0;
    for(size_t n=2;n<=n_target;n++){
        std::vector<std::vector<size_t> > combos=spin_cart_prod(g.vs()[root].rank(),p);
        std::vector<std::vector<size_t> > r_combos=spin_cart_prod(r_k,n);
        std::complex<double> n_res=0;
        for(size_t idx=0;idx<combos.size();idx++){
            std::vector<size_t> c=combos[idx];
            double prob_factor=1;
            for(size_t i=0;i<c.size();i++){
                prob_factor*=g.vs()[root].probs()[c[i]];
            }
            std::complex<double> sub_res=0;
            for(size_t i=0;i<r_combos.size();i++){
                std::complex<double> contrib=observables::q(g,n,n,p,r_combos[i],c,k_components);
                if(abs_flag){
                    sub_res=(std::norm(contrib)>std::norm(sub_res))?contrib:sub_res;
                }
                else{
                    sub_res+=contrib;
                }
            }
            n_res+=sub_res*prob_factor;
        }
        res+=pow(-1,(double) n)*binom(n_target,n_target-n)*pow(q_orig,(double) n-n_target)*pow(g.n_phys_sites(),n_target-n)*n_res;
    }
    //normalize so that perfect correlation is 1
    res+=pow(-1,(double) n_target-1)*(n_target-1)*pow(q_orig,-(double) n_target)*pow(g.n_phys_sites(),n_target);
    res*=pow(q_prefactor(n_target,q_orig),p-1);
    // std::cout<<"final_res: "<<res<<"\n";
    return res;
}
template std::complex<double> observables::q<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,size_t,std::vector<double>,bool);

//this q function calculates using a bottom-up approach. this fixes the stack size issue since all top-down calls resolve with depth 1. this function still requires the top-down version to compute observables as the tree is traversed towards the root.
template<typename cmp>
std::complex<double> observables::q(graph<cmp>& g,size_t n_target,size_t n,size_t p,std::vector<size_t> r,std::vector<size_t> c,std::vector<double> k_components){
    //due to the ordering defined by the comparator, the leaves are at the start and the root is at the end of the multiset.
    for(auto it=g.es().begin();it!=g.es().end();++it){
        std::array<size_t,2> v_idxs{(*it).v1(),(*it).v2()};
        std::vector<size_t> c0(p,0);
        std::vector<size_t> c1(p,0);
        q_known_factors_complex[std::make_tuple(v_idxs[0],n_target,n,p,r,c0,k_components)]=q(g,v_idxs[0],n_target,n,p,r,c0,k_components,0);
        q_known_factors_complex[std::make_tuple(v_idxs[1],n_target,n,p,r,c1,k_components)]=q(g,v_idxs[1],n_target,n,p,r,c1,k_components,0);
    }
    return q(g,g.vs().size()-1,n_target,n,p,r,c,k_components,0);
}
template std::complex<double> observables::q<bmi_comparator>(graph<bmi_comparator>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double>);

template<typename cmp>
void observables::print_moments(graph<cmp>& g,size_t q_orig){ //debug
    double m1_1_abs=observables::m(g,q_orig,g.vs().size()-1,1,1,1);
    double m1_2_abs=observables::m(g,q_orig,g.vs().size()-1,1,2,1);
    // double m1_3_abs=observables::m(g,g.vs().size()-1,1,3,1);
    double m1_1=observables::m(g,q_orig,g.vs().size()-1,1,1,0);
    double m1_2=observables::m(g,q_orig,g.vs().size()-1,1,2,0);
    // double m1_3=observables::m(g,g.vs().size()-1,1,3,0);
    double m2_1=observables::m(g,q_orig,g.vs().size()-1,2,1,0);
    double m2_2=observables::m(g,q_orig,g.vs().size()-1,2,2,0);
    // double m2_3=observables::m(g,g.vs().size()-1,2,3,0);
    double m3_1=observables::m(g,q_orig,g.vs().size()-1,3,1,0);
    double m3_2=observables::m(g,q_orig,g.vs().size()-1,3,2,0);
    // double m3_3=observables::m(g,g.vs().size()-1,3,3,0);
    double m4_1=observables::m(g,q_orig,g.vs().size()-1,4,1,0);
    double m4_2=observables::m(g,q_orig,g.vs().size()-1,4,2,0);
    // double m4_3=observables::m(g,g.vs().size()-1,4,3,0);
    std::cout<<"sum <|si|>: "<<m1_1_abs<<"/"<<g.n_phys_sites()<<" = "<<(m1_1_abs/g.n_phys_sites())<<"\n";
    std::cout<<"sum <|si|>^2: "<<m1_2_abs<<"/"<<g.n_phys_sites()<<" = "<<(m1_2_abs/g.n_phys_sites())<<"\n";
    // std::cout<<"sum <|si|>^3: "<<m1_3_abs<<"/"<<g.n_phys_sites()<<" = "<<(m1_3_abs/g.n_phys_sites())<<"\n";
    std::cout<<"sum <si>: "<<m1_1<<"/"<<g.n_phys_sites()<<" = "<<(m1_1/g.n_phys_sites())<<"\n";
    std::cout<<"sum <si>^2: "<<m1_2<<"/"<<g.n_phys_sites()<<" = "<<(m1_2/g.n_phys_sites())<<"\n";
    // std::cout<<"sum <si>^3: "<<m1_3<<"/"<<g.n_phys_sites()<<" = "<<(m1_3/g.n_phys_sites())<<"\n";
    std::cout<<"sum <si sj>: "<<m2_1<<"/"<<pow(g.n_phys_sites(),2)<<" = "<<(m2_1/pow(g.n_phys_sites(),2))<<"\n";
    std::cout<<"sum <si sj>^2: "<<m2_2<<"/"<<pow(g.n_phys_sites(),2)<<" = "<<(m2_2/pow(g.n_phys_sites(),2))<<"\n";
    // std::cout<<"sum <si sj>^3: "<<m2_3<<"/"<<pow(g.n_phys_sites(),2)<<" = "<<(m2_3/pow(g.n_phys_sites(),2))<<"\n";
    std::cout<<"sum <si sj sk>: "<<m3_1<<"/"<<pow(g.n_phys_sites(),3)<<" = "<<(m3_1/pow(g.n_phys_sites(),3))<<"\n";
    std::cout<<"sum <si sj sk>^2: "<<m3_2<<"/"<<pow(g.n_phys_sites(),3)<<" = "<<(m3_2/pow(g.n_phys_sites(),3))<<"\n";
    // std::cout<<"sum <si sj sk>^3: "<<m3_3<<"/"<<pow(g.n_phys_sites(),3)<<" = "<<(m3_3/pow(g.n_phys_sites(),3))<<"\n";
    std::cout<<"sum <si sj sk sl>: "<<m4_1<<"/"<<pow(g.n_phys_sites(),4)<<" = "<<(m4_1/pow(g.n_phys_sites(),4))<<"\n";
    std::cout<<"sum <si sj sk sl>^2: "<<m4_2<<"/"<<pow(g.n_phys_sites(),4)<<" = "<<(m4_2/pow(g.n_phys_sites(),4))<<"\n";
    // std::cout<<"sum <si sj sk sl>^3: "<<m4_3<<"/"<<pow(g.n_phys_sites(),4)<<" = "<<(m4_3/pow(g.n_phys_sites(),4))<<"\n";
}
template void observables::print_moments<bmi_comparator>(graph<bmi_comparator>&,size_t);

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