#include <fstream>

#include "optimize.hpp"

//DEBUG GLOBALS
size_t pair_opt_ij_count_idx=0;
size_t pair_opt_ki_count_idx=0;
std::ofstream ij_ofs("pair_ij_bmi_dump.txt");
std::ofstream ki_ofs("pair_ki_bmi_dump.txt");
std::ofstream ij_cost_ofs("pair_ij_cost_dump.txt");
std::ofstream ki_cost_ofs("pair_ki_cost_dump.txt");

double vec_add_float(std::vector<double> v){
    std::sort(v.begin(),v.end());
    double sum=0;
    for(size_t i=0;i<v.size();i++){
        sum+=v[i];
    }
    return sum;
}

double vec_mult_float(std::vector<double> v){
    std::sort(v.begin(),v.end());
    double prod=1;
    for(size_t i=0;i<v.size();i++){
        prod*=v[i];
    }
    return prod;
}

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
    std::vector<std::string> ij_bmi_dump_lines;
    std::vector<std::string> ij_cost_dump_lines;
    std::vector<std::string> ki_bmi_dump_lines;
    std::vector<std::string> ki_cost_dump_lines;
    
    //DEBUG OUTPUT
    {
        for(size_t n=0;n<cluster.size();n++){
            std::stringstream ki_bmi_dump_line;
            std::stringstream ki_cost_dump_line;
            ki_bmi_dump_line<<(pair_opt_ki_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<0<<" "<<cluster[n].bmi()<<"\n";
            // std::cout<<ki_eff_k_dump_lines.size()<<"\n";
            ki_bmi_dump_lines.push_back(ki_bmi_dump_line.str());
            //DEBUG: calculate cost function
            double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
            ki_cost_dump_line<<(pair_opt_ki_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<0<<" "<<cost<<"\n";
            ki_cost_dump_lines.push_back(ki_cost_dump_line.str());
        }
        std::stringstream ij_bmi_dump_line;
        std::stringstream ij_cost_dump_line;
        ij_bmi_dump_line<<pair_opt_ij_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<0<<" "<<current.bmi()<<"\n";
        ij_bmi_dump_lines.push_back(ij_bmi_dump_line.str());
        //DEBUG: calculate cost function
        double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
        ij_cost_dump_line<<pair_opt_ij_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<0<<" "<<cost<<"\n";
        ij_cost_dump_lines.push_back(ij_cost_dump_line.str());
    }
    
    //precompute intermediate quantities for weight optimization
    std::vector<double> gi(current.w().nx(),1);
    std::vector<double> gj(current.w().ny(),1);
    array2d<double> pair_ij(current.w().nx(),current.w().ny());
    array2d<double> pair_ij_env(current.w().nx(),current.w().ny());
        
    for(size_t i=0;i<current.w().nx();i++){
        std::vector<double> gi_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){
                // std::cout<<"orig_g1_contrib: "<<(old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v1())?0:1))[i]<<"\n";
                gi_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v1())?1:0))[i]);
            }
        }
        gi[i]=vec_mult_float(gi_factors);
    }
    for(size_t j=0;j<current.w().ny();j++){
        std::vector<double> gj_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v2())||(old_cluster[n].v2()==old_current.v2())){
                // std::cout<<"orig_g2_contrib: "<<(old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v2())?0:1))[j]<<"\n";
                gj_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v2())?1:0))[j]);
            }
        }
        gj[j]=vec_mult_float(gj_factors);
    }
        
    // std::cout<<(std::string) current.f()<<"\n";
    for(size_t n1=0;n1<max_it;n1++){
        std::stringstream ij_bmi_dump_line;
        std::stringstream ij_cost_dump_line;
        //calculate P_{ij},P'^{env}_{ij}
        std::vector<double> sum_pair_ij_addends;
        std::vector<double> sum_pair_ij_env_addends;
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                //calculate G'(f_k(S_i,S_j))
                std::vector<double> gi_prime(r_k,1);
                std::vector<double> gj_prime(r_k,1);
                std::vector<double> gi_prime_factors;
                std::vector<double> gj_prime_factors;
                size_t k=current.f().at(i,j); //select where output is added to
                for(size_t m=0;m<cluster.size();m++){
                    if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                        if(cluster[m].w().ny()>=r_k){ //in case resizing has not yet occured, just ignore if invalid
                            // std::cout<<"gi_prime_contrib: "<<(cluster[m].w().sum_over_axis(0))[k]<<"\n";
                            if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                gi_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                            }
                        }
                    }
                    else{ //connected to site j
                        if(cluster[m].w().ny()>=r_k){ //in case resizing has not yet occured, just ignore if invalid
                            // std::cout<<"gj_prime_contrib: "<<(cluster[m].w().sum_over_axis(0))[k]<<"\n";
                            if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                gj_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                            }
                        }
                    }
                }
                gi_prime[k]=vec_mult_float(gi_prime_factors);
                gj_prime[k]=vec_mult_float(gj_prime_factors);
                
                std::vector<double> factors;
                std::vector<double> factors_env;
                //calculate P_{ij}
                factors.push_back(old_current.w().at(i,j));
                factors.push_back(gi[i]);
                factors.push_back(gj[j]);
                pair_ij.at(i,j)=vec_mult_float(factors);
                sum_pair_ij_addends.push_back(pair_ij.at(i,j));
                //calculate P'^{env}_{ij}
                factors_env.push_back(gi_prime[k]);
                factors_env.push_back(gj_prime[k]);
                pair_ij_env.at(i,j)=vec_mult_float(factors_env);
                sum_pair_ij_env_addends.push_back(pair_ij_env.at(i,j));
            }
        }
        //normalize P_{ij},P'^{env}_{ij}
        double sum_pair_ij=vec_add_float(sum_pair_ij_addends);
        double sum_pair_ij_env=vec_add_float(sum_pair_ij_env_addends);
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                if(fabs(sum_pair_ij)>1e-10){
                    pair_ij.at(i,j)/=sum_pair_ij;
                }
                if(fabs(sum_pair_ij_env)>1e-10){
                    pair_ij_env.at(i,j)/=sum_pair_ij_env;
                }
            }
        }
        // std::cout<<(std::string)current.f()<<"\n";
        // std::cout<<(std::string)pair_ij<<"\n";
        // std::cout<<(std::string)pair_ij_env<<"\n";
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                    if(fabs(pair_ij_env.at(i,j))>1e-10){
                        current.w().at(i,j)=pair_ij.at(i,j)/pair_ij_env.at(i,j);
                    }
                }
                else{ //lr!=0 means use gradient descent with lr
                    if(fabs(current.w().at(i,j))>1e-10){
                        current.w().at(i,j)-=lr*((pair_ij_env.at(i,j)*current.w().at(i,j))-pair_ij.at(i,j))/current.w().at(i,j);
                    }
                }
            }
        }
        // std::cout<<"prefinal current.w():\n"<<(std::string)current.w()<<"\n";
        current.j(current.w().nx(),current.w()); //only valid for potts models with constant rank
        current.bmi(current.w());
        // std::cout<<"final current.w():\n"<<(std::string)current.w()<<"\n";
        
        //DISABLE if using f_alg1, CAUSES INSTABILITY IN OPTIMIZATION
        // current.f()=optimize::f_maxent(sites[current.v1()],sites[current.v2()],current.w(),r_k);
        // std::cout<<(std::string) current.w()<<"\n";
        // std::cout<<(std::string) current.f()<<"\n";
        
        //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
        //pair_ki,imu if source==current.v1(), pair_kj,jnu if source==current.v2(). wlog, use pair_ki and imu.
        //k, the new index, is always second because virtual indices > physical indices.
        for(size_t n=0;n<cluster.size();n++){
            //determine whether this bond was connected to site i or to site j
            size_t source;
            if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
                source=(old_cluster[n].v1()==master)?master:slave;
            }
            else{
                source=(old_cluster[n].v2()==master)?master:slave;
            }
        
            std::stringstream ki_bmi_dump_line;
            std::stringstream ki_cost_dump_line;
            array2d<double> pair_ki(cluster[n].w().nx(),cluster[n].w().ny());
            array2d<double> pair_ki_env(cluster[n].w().nx(),cluster[n].w().ny());
            //reinitialize weight matrix to correct size if needed on first iteration
            if(n1==0){ //if first iteration, resize pair_ki
                if(source==current.v1()){
                    pair_ki=array2d<double>((current.v1()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
                    pair_ki_env=array2d<double>((current.v1()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
                }
                else{
                    pair_ki=array2d<double>((current.v2()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
                    pair_ki_env=array2d<double>((current.v2()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
                }
            }
            if((pair_ki.nx()!=cluster[n].w().nx())||(pair_ki.ny()!=cluster[n].w().ny())){
                // std::cout<<(std::string)cluster[n].w()<<"\n";
                array2d<double> new_w(pair_ki.nx(),pair_ki.ny());
                for(size_t i=0;i<new_w.nx();i++){
                    for(size_t j=0;j<new_w.ny();j++){
                        if((i<cluster[n].w().nx())&&(j<cluster[n].w().ny())){
                            new_w.at(i,j)=cluster[n].w().at(i,j);
                        }
                        else{
                            new_w.at(i,j)=0;
                        }
                    }
                }
                cluster[n].w()=new_w;
                // std::cout<<(std::string)cluster[n].w()<<"\n";
            }
            //prep to calculate P^_{ki_\mu}, P'^{env}_{ki_\mu}
            std::vector<std::vector<double> > addends(pair_ki.nx()*pair_ki.ny());
            std::vector<std::vector<double> > addends_env(pair_ki_env.nx()*pair_ki_env.ny());
            std::vector<double> sum_pair_ki_addends;
            std::vector<double> sum_pair_ki_env_addends;
            cluster[n].bmi(cluster[n].w());
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    //calculate G'(f_k(S_i,S_j))
                    std::vector<double> gi_prime(r_k,1);
                    std::vector<double> gj_prime(r_k,1);
                    std::vector<double> gi_prime_factors;
                    std::vector<double> gj_prime_factors;
                    size_t k=current.f().at(i,j); //select where output is added to
                    for(size_t m=0;m<cluster.size();m++){
                        if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                            if(cluster[m].w().ny()>=r_k){ //in case resizing has not yet occured, just ignore if invalid
                                // std::cout<<"gi_prime_contrib: "<<(cluster[m].w().sum_over_axis(0))[k]<<"\n";
                                if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                    gi_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                                }
                            }
                        }
                        else{ //connected to site j
                            if(cluster[m].w().ny()>=r_k){ //in case resizing has not yet occured, just ignore if invalid
                                // std::cout<<"gj_prime_contrib: "<<(cluster[m].w().sum_over_axis(0))[k]<<"\n";
                                if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                    gj_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                                }
                            }
                        }
                    }
                    gi_prime[k]=vec_mult_float(gi_prime_factors);
                    gj_prime[k]=vec_mult_float(gj_prime_factors);
                    //divide appropriate factor
                    if(source==current.v1()){
                        if(fabs((cluster[n].w().sum_over_axis(0))[k])>1e-10){
                            gi_prime[k]/=(cluster[n].w().sum_over_axis(0))[k]; //always ax0 because ax1 always refers to new virtual site
                        }
                    }
                    else{
                        if(fabs((cluster[n].w().sum_over_axis(0))[k])>1e-10){
                            gj_prime[k]/=(cluster[n].w().sum_over_axis(0))[k]; //always ax0 because ax1 always refers to new virtual site
                        }
                    }
                    //get addends
                    for(size_t imu=0;imu<pair_ki.nx();imu++){
                        std::vector<double> factors;
                        std::vector<double> factors_env;
                        //calculate P_{ki_\mu} addends
                        factors.push_back(old_current.w().at(i,j));
                        if(source==current.v1()){ //connected to site i
                            if(old_cluster[n].v1()==current.v1()){ //site imu > site i
                                factors.push_back(old_cluster[n].w().at(i,imu));
                            }
                            else{ //site imu < site i
                                factors.push_back(old_cluster[n].w().at(imu,i));
                            }
                        }
                        else{ //connected to site j
                            if(old_cluster[n].v1()==current.v2()){ //site imu > site j
                                factors.push_back(old_cluster[n].w().at(j,imu));
                            }
                            else{ //site imu < site j
                                factors.push_back(old_cluster[n].w().at(imu,j));
                            }
                        }
                        factors.push_back(gi[i]);
                        factors.push_back(gj[j]);
                        sum_pair_ki_addends.push_back(vec_mult_float(factors));
                        addends[(k*pair_ki.nx())+imu].push_back(vec_mult_float(factors));
                        //calculate P'^{env}_{ki_\mu} addends
                        factors_env.push_back(current.w().at(i,j));
                        factors_env.push_back(gi_prime[k]);
                        factors_env.push_back(gj_prime[k]);
                        sum_pair_ki_env_addends.push_back(vec_mult_float(factors_env));
                        addends_env[(k*pair_ki_env.nx())+imu].push_back(vec_mult_float(factors_env));
                    }
                }
            }
            //calculate P^_{ki_\mu}, P'^{env}_{ki_\mu}
            for(size_t imu=0;imu<pair_ki.nx();imu++){
                for(size_t k=0;k<pair_ki.ny();k++){
                    pair_ki.at(imu,k)=vec_add_float(addends[(k*pair_ki.nx())+imu]);
                    pair_ki_env.at(imu,k)=vec_add_float(addends_env[(k*pair_ki.nx())+imu]);
                }
            }
            double sum_pair_ki=vec_add_float(sum_pair_ki_addends);
            double sum_pair_ki_env=vec_add_float(sum_pair_ki_env_addends);
            //normalize P_{ki_\mu},P'^{env}_{ki_\mu}
            for(size_t i=0;i<pair_ki.nx();i++){
                for(size_t j=0;j<pair_ki.ny();j++){
                    if(fabs(sum_pair_ki)>1e-10){
                        pair_ki.at(i,j)/=sum_pair_ki;
                    }
                    if(fabs(sum_pair_ki_env)>1e-10){
                        pair_ki_env.at(i,j)/=sum_pair_ki_env;
                    }
                }
            }
            // std::cout<<"pre-reinit f:\n"<<(std::string)current.f()<<"\n";
            // std::cout<<"pre-reinit pair_ki:\n"<<(std::string)pair_ki<<"\n";
            // std::cout<<"pre-reinit pair_ki_env:\n"<<(std::string)pair_ki_env<<"\n";
            
            for(size_t i=0;i<pair_ki.nx();i++){
                for(size_t j=0;j<pair_ki.ny();j++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        if(fabs(pair_ki_env.at(i,j))>1e-10){
                            cluster[n].w().at(i,j)=pair_ki.at(i,j)/pair_ki_env.at(i,j);
                        }
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        if(fabs(cluster[n].w().at(i,j))>1e-10){
                            cluster[n].w().at(i,j)-=lr*((pair_ki_env.at(i,j)*cluster[n].w().at(i,j))-pair_ki.at(i,j))/cluster[n].w().at(i,j);
                        }
                    }
                }
            }
            // std::cout<<"prefinal cluster[n].w():\n"<<(std::string)cluster[n].w()<<"\n";
            cluster[n].j(cluster[n].w().nx(),cluster[n].w()); //only valid for potts models with constant rank
            cluster[n].bmi(cluster[n].w());
            // std::cout<<"postfinal cluster[n].w():\n"<<(std::string)cluster[n].w()<<"\n";
            
            //DEBUG OUTPUT
            ki_bmi_dump_line<<(pair_opt_ki_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cluster[n].bmi()<<"\n";
            // std::cout<<ki_eff_k_dump_lines.size()<<"\n";
            ki_bmi_dump_lines.push_back(ki_bmi_dump_line.str());
            //DEBUG: calculate cost function
            double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
            ki_cost_dump_line<<(pair_opt_ki_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
            ki_cost_dump_lines.push_back(ki_cost_dump_line.str());
        }
        
        //DEBUG OUTPUT
        ij_bmi_dump_line<<pair_opt_ij_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<current.bmi()<<"\n";
        ij_bmi_dump_lines.push_back(ij_bmi_dump_line.str());
        //DEBUG: calculate cost function
        double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
        ij_cost_dump_line<<pair_opt_ij_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
        ij_cost_dump_lines.push_back(ij_cost_dump_line.str());
    }
    pair_opt_ij_count_idx++;
    for(size_t i=0;i<ij_bmi_dump_lines.size();i++){
        ij_ofs<<ij_bmi_dump_lines[i];
        ij_cost_ofs<<ij_cost_dump_lines[i];
    }
    pair_opt_ki_count_idx+=cluster.size();
    for(size_t i=0;i<ki_bmi_dump_lines.size();i++){
        ki_ofs<<ki_bmi_dump_lines[i];
        ki_cost_ofs<<ki_cost_dump_lines[i];
    }
    // std::cout<<(std::string) current.w()<<"\n";
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
        size_t k=std::distance(w_sums.begin(),std::min_element(w_sums.begin(),w_sums.end())); //choose first case of smallest cumulative sum
        w_sums[k]+=max;
        // std::cout<<v_i.vol()<<" "<<v_j.vol()<<" "<<i<<" "<<j<<" "<<k<<" "<<w_sums[k]<<"\n";
        w.at(i,j)=-1; //weights can never be negative so this element will never be the max. also, w is passed by value so no change to original.
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
    // std::cout<<(std::string)current.f()<<"\n";
    // std::cout<<(std::string)current.w()<<"\n";
    
    array2d<double> g(sites[current.v1()].rank(),sites[current.v2()].rank());
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double g_res=1;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum=0;
                size_t source;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                    source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
                }
                else{
                    source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
                }
                if(source==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            spin_sum+=old_cluster[n].w().at(i,imu);
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            spin_sum+=old_cluster[n].w().at(imu,i);
                        }
                    }
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            spin_sum+=old_cluster[n].w().at(j,imu);
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            spin_sum+=old_cluster[n].w().at(imu,j);
                        }
                    }
                }
                g_res*=spin_sum;
            }
            g.at(i,j)=g_res;
        }
    }
    
    double a_old=0;
    double b_old=0;
    for(size_t i=0;i<sites[old_current.v1()].rank();i++){
        for(size_t j=0;j<sites[old_current.v2()].rank();j++){
            double contrib_a=(old_current.w().at(i,j)>0)?old_current.w().at(i,j)*g.at(i,j)*log(old_current.w().at(i,j)):old_current.w().at(i,j)*g.at(i,j)*log(1e-100);
            double contrib_b=old_current.w().at(i,j)*g.at(i,j);
            double contrib_b_logs=0;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum_z_prime=0;
                double spin_sum_b=0;
                size_t source;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                    source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
                }
                else{
                    source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
                }
                if(source==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(old_cluster[n].w().at(i,imu)>0){
                                spin_sum_b+=log(old_cluster[n].w().at(i,imu));
                            }
                            else{
                                spin_sum_b+=log(1e-100);
                            }
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(old_cluster[n].w().at(imu,i)>0){
                                spin_sum_b+=log(old_cluster[n].w().at(imu,i));
                            }
                            else{
                                spin_sum_b+=log(1e-100);
                            }
                        }
                    }
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(old_cluster[n].w().at(j,imu)>0){
                                spin_sum_b+=log(old_cluster[n].w().at(j,imu));
                            }
                            else{
                                spin_sum_b+=log(1e-100);
                            }
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(old_cluster[n].w().at(imu,j)>0){
                                spin_sum_b+=log(old_cluster[n].w().at(imu,j));
                            }
                            else{
                                spin_sum_b+=log(1e-100);
                            }
                        }
                    }
                }
                contrib_b_logs+=spin_sum_b;
            }
            a_old+=contrib_a;
            b_old+=contrib_b*contrib_b_logs;
        }
    }
    
    double z_prime=0;
    double z=0;
    double a=0;
    double b=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib_z_prime=current.w().at(i,j);
            double contrib_z=old_current.w().at(i,j)*g.at(i,j);
            // double contrib_a=(current.w().at(i,j)>0)?old_current.w().at(i,j)*g.at(i,j)*log(current.w().at(i,j)):old_current.w().at(i,j)*g.at(i,j)*log(1e-100);
            double contrib_a=(current.w().at(i,j)>0)?old_current.w().at(i,j)*g.at(i,j)*log(current.w().at(i,j)):0;
            double contrib_b=old_current.w().at(i,j)*g.at(i,j);
            size_t k=current.f().at(i,j);
            double contrib_b_logs=0;
            for(size_t n=0;n<cluster.size();n++){
                double spin_sum_z_prime=0;
                double spin_sum_b=0;
                size_t source;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                    source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
                }
                else{
                    source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
                }
                if(source==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_b+=log(cluster[n].w().at(imu,k));
                            }
                            // else{
                                // spin_sum_b+=log(1e-100);
                            // }
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_b+=log(cluster[n].w().at(imu,k));
                            }
                            // else{
                                // spin_sum_b+=log(1e-100);
                            // }
                        }
                    }
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_b+=log(cluster[n].w().at(imu,k));
                            }
                            // else{
                                // spin_sum_b+=log(1e-100);
                            // }
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_b+=log(cluster[n].w().at(imu,k));
                            }
                            // else{
                                // spin_sum_b+=log(1e-100);
                            // }
                        }
                    }
                }
                for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                    spin_sum_z_prime+=cluster[n].w().at(imu,k);
                }
                contrib_z_prime*=spin_sum_z_prime;
                contrib_b_logs+=spin_sum_b;
            }
            z_prime+=contrib_z_prime;
            z+=contrib_z;
            a+=contrib_a;
            b+=contrib_b*contrib_b_logs;
        }
    }
    // std::cout<<"z: "<<z<<"\nz': "<<z_prime<<"\n";
    // std::cout<<"a: "<<a<<"\nb: "<<b<<"\n";
    // double res=log(z_prime)-(a/z);
    // double res=(a/z)+(b/z)-log(z_prime);
    // double res=log(z_prime)-(a/z)-(b/z);
    // double res=(a_old/z)+(b_old/z)-log(z);
    double res=((a_old/z)+(b_old/z)-log(z))-((a/z)+(b/z)-log(z_prime));
    return res;
}