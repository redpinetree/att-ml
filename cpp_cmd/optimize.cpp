#include <fstream>

#include "optimize.hpp"

//DEBUG GLOBALS
size_t pair_opt_count_idx=0;
std::ofstream ij_ofs("pair_ij_eff_k_dump.txt");
std::ofstream ki_ofs("pair_ki_eff_k_dump.txt");
std::ofstream ij_cost_ofs("pair_ij_cost_dump.txt");
std::ofstream ki_cost_ofs("pair_ki_cost_dump.txt");

void optimize::potts_renorm(size_t slave,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster){
    size_t q=current.w().nx(); //assumption is that rank(=q) is constant!
    for(size_t n=0;n<cluster.size();n++){
        if((old_cluster[n].v1()==slave)||(old_cluster[n].v2()==slave)){
            cluster[n].j()=renorm_coupling(q,cluster[n].j(),current.j()); //update bond weight
            for(size_t i=0;i<q;i++){
                for(size_t j=0;j<q;j++){
                    cluster[n].w().at(i,j)=(i==j)?(1/(q+(q*(q-1)*exp(-cluster[n].j())))):(1/((q*exp(cluster[n].j()))+(q*(q-1))));
                }
            }
            cluster[n].j(q,cluster[n].w());
            cluster[n].bmi(cluster[n].w());
        }
    }
}

//debug ver
//TODO: make consistent ndarray addressing like (x,y) (currently, (y,x))
//TODO: add convergence criteria for both iterative eq and grad desc methods
//TODO: move weight matrix reinit to dedicated function in ndarray (like expand_dims)
void optimize::kl_iterative(size_t master,size_t slave,size_t r_k,std::vector<site> sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,double lr=0){
    //precompute intermediate quantities for weight optimization
    std::vector<double> orig_g1(current.w().nx(),1);
    std::vector<double> orig_g2(current.w().ny(),1);
    std::vector<double> g1(current.w().nx(),1);
    std::vector<double> g2(current.w().ny(),1);
    array2d<double> pair_ij(current.w().ny(),current.w().nx());
    array2d<double> pair_ij_env(current.w().ny(),current.w().nx());
    
    for(size_t i=0;i<current.w().nx();i++){
        orig_g1[i]=1;
        for(auto it=sites[current.v1()].adj().begin();it!=sites[current.v1()].adj().end();++it){
            // std::cout<<"g1_contrib: "<<((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i]<<"\n";
            orig_g1[i]*=((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i];
        }
    }
    for(size_t j=0;j<current.w().ny();j++){
        orig_g2[j]=1;
        for(auto it=sites[current.v2()].adj().begin();it!=sites[current.v2()].adj().end();++it){
            // std::cout<<"g2_contrib: "<<((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[i]<<"\n";
            orig_g2[j]*=((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j];
        }
    }
    
    std::vector<std::string> ij_eff_k_dump_lines;
    std::vector<std::string> ij_cost_dump_lines;
    // std::cout<<"IJ opt for bond between sites ("<<current.v1()<<","<<current.v2()<<")\n";
    double sum_pair_ij=0;
    for(size_t i=0;i<pair_ij.ny();i++){
        for(size_t j=0;j<pair_ij.nx();j++){
            pair_ij.at(i,j)=current.w().at(i,j)*orig_g1[i]*orig_g2[j];
            sum_pair_ij+=pair_ij.at(i,j);
        }
    }
    
    for(size_t n1=0;n1<max_it;n1++){
        std::stringstream ij_eff_k_dump_line;
        std::stringstream ij_cost_dump_line;
        //calculate pair_{ij}
        //TODO: calculate pair'_{ij}^{env} (uses cm func), can be reused for pair'_{ki_mu}^{env}?
        double sum_pair_ij_env=0;
        for(size_t i=0;i<pair_ij.ny();i++){
            for(size_t j=0;j<pair_ij.nx();j++){
                double g1_env=1;
                double g2_env=1;
                size_t k=current.f().at(i,j); //select where output is added to
                for(auto it=sites[current.v1()].adj().begin();it!=sites[current.v1()].adj().end();++it){
                    g1_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                }
                for(auto it=sites[current.v2()].adj().begin();it!=sites[current.v2()].adj().end();++it){
                    g2_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                }
                pair_ij_env.at(i,j)=g1_env*g2_env;
                sum_pair_ij_env+=current.w().at(i,j)*pair_ij_env.at(i,j);
            }
        }
        for(size_t i=0;i<pair_ij.ny();i++){
            for(size_t j=0;j<pair_ij.nx();j++){
                pair_ij.at(i,j)/=sum_pair_ij;
                pair_ij_env.at(i,j)/=sum_pair_ij_env;
            }
        }
        for(size_t i=0;i<pair_ij.ny();i++){
            for(size_t j=0;j<pair_ij.nx();j++){
                if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                    current.w().at(i,j)=pair_ij.at(i,j)/pair_ij_env.at(i,j);
                }
                else{ //lr!=0 means use gradient descent with lr
                    current.w().at(i,j)-=lr*((pair_ij_env.at(i,j)*current.w().at(i,j))-pair_ij.at(i,j))/current.w().at(i,j);
                }
            }
        }
        current.j(current.w().nx(),current.w()); //only valid for potts models with constant rank
        current.bmi(current.w());
        
        //DEBUG OUTPUT
        ij_eff_k_dump_line<<pair_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<current.j()<<"\n";
        ij_eff_k_dump_lines.push_back(ij_eff_k_dump_line.str());
        //DEBUG: calculate cost function
        double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
        ij_cost_dump_line<<pair_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
        ij_cost_dump_lines.push_back(ij_cost_dump_line.str());
    }
    pair_opt_count_idx++;
    for(size_t i=0;i<ij_eff_k_dump_lines.size();i++){
        ij_ofs<<ij_eff_k_dump_lines[i];
        ij_cost_ofs<<ij_cost_dump_lines[i];
    }
    //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
    //pair_ki,imu if source==current.v1(), pair_kj,jnu if source==current.v2(). wlog, use pair_ki and imu.
    //k, the new index, is always second because virtual indices > physical indices.
    //TODO: fix for variable r_k...
    // std::cout<<"HERE kl_iterative\n";
    for(size_t n=0;n<cluster.size();n++){
        // std::cout<<"KI opt for bond between sites ("<<cluster[n].v1()<<","<<cluster[n].v2()<<")\n";
        std::vector<std::string> ki_eff_k_dump_lines;
        std::vector<std::string> ki_cost_dump_lines;
        
        for(size_t n2=0;n2<max_it;n2++){
            // std::cout<<"HERE opt\n";
            size_t source;
            if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
                source=(old_cluster[n].v1()==master)?master:slave;
            }
            else{
                source=(old_cluster[n].v2()==master)?master:slave;
            }
            std::stringstream ki_eff_k_dump_line;
            std::stringstream ki_cost_dump_line;
            array2d<double> pair_ki(r_k,(source==current.v1())?cluster[n].w().nx():cluster[n].w().ny());
            array2d<double> pair_ki_env(r_k,(source==current.v1())?cluster[n].w().nx():cluster[n].w().ny());
            double sum_pair_ki=0;
            double sum_pair_ki_env=0;
            std::vector<std::vector<double> > addends(pair_ki.nx()*pair_ki.ny());
            std::vector<std::vector<double> > addends_env(pair_ki_env.nx()*pair_ki_env.ny());
            for(size_t i=0;i<cluster[n].w().ny();i++){
                for(size_t j=0;j<cluster[n].w().nx();j++){
                    size_t k=current.f().at(i,j); //select where output is added to
                    double orig_g1_ki=orig_g1[i];
                    double orig_g2_ki=orig_g2[j];
                    double g1_ki_env=g1[i];
                    double g2_ki_env=g2[j];
                    if(source==current.v1()){
                        orig_g1_ki/=(old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==current.v1())?0:1))[i];
                        g1_ki_env/=(cluster[n].w().sum_over_axis((cluster[n].v1()==current.v1())?0:1))[i];
                    }
                    else{
                        orig_g2_ki/=(old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==current.v2())?0:1))[j];
                        g2_ki_env/=(cluster[n].w().sum_over_axis((cluster[n].v1()==current.v2())?0:1))[j];
                    }
                    // std::cout<<"g1*g2: "<<(g1_ki*g2_ki)<<"\n";
                    // std::cout<<std::string(current.w());
                    for(size_t imu=0;imu<pair_ki.nx();imu++){
                        //TODO: how to reinitialize cluster[n].w() for r_max>q? current iteration appears to worsen the approximation...
                        // std::cout<<imu<<" "<<k<<" "<<pair_ki.nx()<<" "<<addends.size()<<" "<<((k*pair_ki.nx())+imu)<<"\n";
                        // std::cout<<imu<<" "<<k<<" "<<pair_ki_env.nx()<<" "<<addends_env.size()<<" "<<((k*pair_ki_env.nx())+imu)<<"\n";
                        addends[(k*pair_ki.nx())+imu].push_back(old_current.w().at(i,j)*old_cluster[n].w().at(imu,(source==current.v1())?i:j)*orig_g1_ki*orig_g2_ki);
                        addends_env[(k*pair_ki_env.nx())+imu].push_back(current.w().at(i,j)*g1_ki_env*g2_ki_env);
                        sum_pair_ki+=old_current.w().at(i,j)*orig_g1_ki*orig_g2_ki;
                        sum_pair_ki_env+=current.w().at(i,j)*g1_ki_env*g2_ki_env;
                    }
                }
            }
            for(size_t imu=0;imu<pair_ki.ny();imu++){
                for(size_t k=0;k<pair_ki.nx();k++){
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
            for(size_t i=0;i<pair_ki.ny();i++){
                for(size_t j=0;j<pair_ki.nx();j++){
                    pair_ki.at(i,j)/=sum_pair_ki;
                    pair_ki_env.at(i,j)/=sum_pair_ki_env;
                }
            }
            //TODO: if rank increased, reinitialize cluster[n].w() with new correct size.
            // std::cout<<"HERE reinit rank\n";
            if((pair_ki.nx()!=cluster[n].w().nx())||(pair_ki.ny()!=cluster[n].w().ny())){
                array2d<double> new_w(pair_ki.ny(),pair_ki.nx());
                for(size_t i=0;i<cluster[n].w().ny();i++){
                    for(size_t j=0;j<cluster[n].w().nx();j++){
                        new_w.at(i,j)=cluster[n].w().at(i,j);
                    }
                }
                cluster[n].w()=new_w;
                cluster[n].f()=optimize::f_maxent(sites[cluster[n].v1()],sites[cluster[n].v2()],cluster[n].w(),r_k);
            }
            // std::cout<<"HERE reinit rank done\n";
            for(size_t i=0;i<pair_ki.ny();i++){
                for(size_t j=0;j<pair_ki.nx();j++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        cluster[n].w().at(i,j)=pair_ki.at(i,j)/pair_ki_env.at(i,j);
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        cluster[n].w().at(i,j)-=0.1*((pair_ki_env.at(i,j)*cluster[n].w().at(i,j))-pair_ki.at(i,j))/cluster[n].w().at(i,j);
                    }
                }
            }
            cluster[n].j(cluster[n].w().nx(),cluster[n].w()); //only valid for potts models with constant rank
            cluster[n].bmi(cluster[n].w());
            
            //DEBUG OUTPUT
            ki_eff_k_dump_line<<pair_opt_count_idx<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n2+1)<<" "<<cluster[n].j()<<"\n";
            ki_eff_k_dump_lines.push_back(ki_eff_k_dump_line.str());
            //DEBUG: calculate cost function
            double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
            ki_cost_dump_line<<pair_opt_count_idx<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n2+1)<<" "<<cost<<"\n";
            ki_cost_dump_lines.push_back(ki_cost_dump_line.str());
            // std::cout<<"HERE opt done\n";
        }
        pair_opt_count_idx++;
        for(size_t i=0;i<ki_eff_k_dump_lines.size();i++){
            ki_ofs<<ki_eff_k_dump_lines[i];
            ki_cost_ofs<<ki_cost_dump_lines[i];
        }
    }
    // std::cout<<"HERE kl_iterative done\n";
}

//cmd function mimicking alg. 1. this assumes and results in a fixed rank.
array2d<size_t> optimize::f_alg1(site v_i,site v_j){
    array2d<size_t> f_res(v_i.rank(),v_j.rank());
    for(size_t s_i=0;s_i<v_i.rank();s_i++){
        for(size_t s_j=0;s_j<v_j.rank();s_j++){
            f_res.at(s_i,s_j)=(v_i.vol()>=v_j.vol())?s_i:s_j;
        }
    }
    return f_res;
}

//cmd function that maximizes the entropy
array2d<size_t> optimize::f_maxent(site v_i,site v_j,array2d<double> w,size_t r_k){
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
    // double S=0;
    // for(size_t n=0;n<w_sums.size();n++){
        // S-=w_sums[n]*log(w_sums[n]);
    // }
    // std::cout<<"entropy S="<<S<<"\n";
    return f_res;
}

double optimize::renorm_coupling(size_t q,double k1,double k2){
    double num=exp(k1+k2)+(q-1);
    double denom=exp(k1)+exp(k2)+(q-2);
    return log(num/denom);
}

double optimize::kl_div(std::vector<site> sites,bond old_current,std::vector<bond> old_cluster,bond current,std::vector<bond> cluster){
    double res=0;
    double z=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib=old_current.w().at(i,j);
            for(size_t n=0;n<cluster.size();n++){
                size_t site_idx=((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].v2():old_cluster[n].v1();
                double spin_sum=0;
                for(size_t imu=0;imu<sites[site_idx].rank();imu++){
                    spin_sum+=(site_idx==old_cluster[n].v2())?old_cluster[n].w().at(i,imu):old_cluster[n].w().at(imu,i);
                }
                contrib*=spin_sum;
            }
            z+=contrib;
        }
    }
    double z_prime=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib=current.w().at(i,j);
            for(size_t n=0;n<cluster.size();n++){
                size_t site_idx=((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].v2():old_cluster[n].v1();
                double spin_sum=0;
                for(size_t imu=0;imu<sites[site_idx].rank();imu++){
                    spin_sum+=cluster[n].w().at(imu,current.f().at(i,j));
                }
                contrib*=spin_sum;
            }
            z_prime+=contrib;
        }
    }
    double a=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib=old_current.w().at(i,j)*log(current.w().at(i,j));
            for(size_t n=0;n<cluster.size();n++){
                size_t site_idx=((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].v2():old_cluster[n].v1();
                double spin_sum=0;
                for(size_t imu=0;imu<sites[site_idx].rank();imu++){
                    spin_sum+=(site_idx==old_cluster[n].v2())?old_cluster[n].w().at(i,imu):old_cluster[n].w().at(imu,i);
                }
                contrib*=spin_sum;
            }
            a+=contrib;
        }
    }
    a/=z;
    double b=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib=old_current.w().at(i,j);
            for(size_t n=0;n<cluster.size();n++){
                size_t site_idx=((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].v2():old_cluster[n].v1();
                double spin_sum=0;
                for(size_t imu=0;imu<sites[site_idx].rank();imu++){
                    spin_sum+=(site_idx==old_cluster[n].v2())?old_cluster[n].w().at(i,imu)*log(current.w().at(imu,current.f().at(i,j))):old_cluster[n].w().at(imu,i)*log(current.w().at(imu,current.f().at(i,j)));
                }
                contrib*=spin_sum;
            }
            b+=contrib;
        }
    }
    b/=z;
    // std::cout<<"z: "<<z<<"\nz': "<<z_prime<<"\n";
    // std::cout<<"a: "<<a<<"\nb: "<<b<<"\n";
    res=a+b-log(z_prime);
    return res;
}
