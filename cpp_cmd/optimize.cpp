#include <fstream>

#include "optimize.hpp"

//DEBUG GLOBALS
/* size_t p_prime_ij_opt_count_idx=0;
size_t p_prime_ki_opt_count_idx=0;
std::ofstream ij_ofs("pair_ij_bmi_dump.txt");
std::ofstream ki_ofs("pair_ki_bmi_dump.txt");
std::ofstream ij_cost_ofs("pair_ij_cost_dump.txt");
std::ofstream ki_cost_ofs("pair_ki_cost_dump.txt");
 */
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
    /* std::vector<std::string> ij_bmi_dump_lines;
    std::vector<std::string> ij_cost_dump_lines;
    std::vector<std::string> ki_bmi_dump_lines;
    std::vector<std::string> ki_cost_dump_lines;
    
    //DEBUG OUTPUT
    {
        for(size_t n=0;n<cluster.size();n++){
            std::stringstream ki_bmi_dump_line;
            std::stringstream ki_cost_dump_line;
            ki_bmi_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<0<<" "<<cluster[n].bmi()<<"\n";
            // std::cout<<ki_eff_k_dump_lines.size()<<"\n";
            ki_bmi_dump_lines.push_back(ki_bmi_dump_line.str());
            //DEBUG: calculate cost function
            double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
            ki_cost_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<0<<" "<<cost<<"\n";
            ki_cost_dump_lines.push_back(ki_cost_dump_line.str());
        }
        std::stringstream ij_bmi_dump_line;
        std::stringstream ij_cost_dump_line;
        ij_bmi_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<0<<" "<<current.bmi()<<"\n";
        ij_bmi_dump_lines.push_back(ij_bmi_dump_line.str());
        //DEBUG: calculate cost function
        double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
        ij_cost_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<0<<" "<<cost<<"\n";
        ij_cost_dump_lines.push_back(ij_cost_dump_line.str());
    } */
    
    //reinitialize weight matrix to correct size if needed on first iteration
    for(size_t n=0;n<cluster.size();n++){
        array2d<double> p_ki(cluster[n].w().nx(),cluster[n].w().ny());
        array2d<double> p_prime_ki_env(cluster[n].w().nx(),cluster[n].w().ny());
        //determine whether this bond was connected to site i or to site j
        size_t source;
        if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
            source=(old_cluster[n].v1()==master)?master:slave;
        }
        else{
            source=(old_cluster[n].v2()==master)?master:slave;
        }
        { //if first iteration, resize p_prime_ki
            if(source==current.v1()){
                p_ki=array2d<double>((current.v1()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
                p_prime_ki_env=array2d<double>((current.v1()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
            }
            else{
                p_ki=array2d<double>((current.v2()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
                p_prime_ki_env=array2d<double>((current.v2()==old_cluster[n].v1())?cluster[n].w().ny():cluster[n].w().nx(),r_k);
            }
        }
        if((p_ki.nx()!=cluster[n].w().nx())||(p_ki.ny()!=cluster[n].w().ny())){
            array2d<double> new_w(p_ki.nx(),p_ki.ny());
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    if((i<cluster[n].w().nx())&&(j<cluster[n].w().ny())){
                        new_w.at(i,j)=cluster[n].w().at(i,j)*(cluster[n].w().nx()*cluster[n].w().ny())/(double) (new_w.nx()*new_w.ny());
                        // new_w.at(i,j)=cluster[n].w().at(i,j);
                    }
                    else{
                        new_w.at(i,j)=1/(double) (new_w.nx()*new_w.ny());
                        // new_w.at(i,j)=0;
                    }
                }
            }
            cluster[n].w()=new_w;
            cluster[n].bmi(cluster[n].w());
            // std::cout<<(std::string) new_w<<"\n";
        }
    }
    
    //precompute intermediate quantities for weight optimization
    std::vector<double> gi(old_current.w().nx(),1);
    std::vector<double> gj(old_current.w().ny(),1);
    array2d<double> p_ij(current.w().nx(),current.w().ny());
    array2d<double> p_prime_ij_env(current.w().nx(),current.w().ny());
        
    for(size_t i=0;i<old_current.w().nx();i++){
        std::vector<double> gi_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){
                gi_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v1())?1:0))[i]);
            }
        }
        gi[i]=vec_mult_float(gi_factors);
    }
    for(size_t j=0;j<old_current.w().ny();j++){
        std::vector<double> gj_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v2())||(old_cluster[n].v2()==old_current.v2())){
                gj_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v2())?1:0))[j]);
            }
        }
        gj[j]=vec_mult_float(gj_factors);
    }
    for(size_t n1=0;n1<max_it;n1++){
        /* std::stringstream ij_bmi_dump_line;
        std::stringstream ij_cost_dump_line; */
        // std::cout<<"old current.w():\n"<<(std::string)current.w()<<"\n";
        //calculate P_{ij},P'^{env}_{ij}
        std::vector<double> sum_p_ij_addends;
        std::vector<double> sum_p_prime_ij_env_addends;
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                //calculate G'(f_k(S_i,S_j))
                std::vector<double> gi_prime(r_k,1);
                std::vector<double> gj_prime(r_k,1);
                std::vector<double> gi_prime_factors;
                std::vector<double> gj_prime_factors;
                size_t k=current.f().at(i,j); //select where output is added to
                for(size_t m=0;m<cluster.size();m++){
                    if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                        if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                            gi_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                        }
                    }
                    else{ //connected to site j
                        if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                            gj_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
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
                p_ij.at(i,j)=vec_mult_float(factors);
                sum_p_ij_addends.push_back(p_ij.at(i,j));
                //calculate P'^{env}_{ij}
                factors_env.push_back(gi_prime[k]);
                factors_env.push_back(gj_prime[k]);
                p_prime_ij_env.at(i,j)=vec_mult_float(factors_env);
                sum_p_prime_ij_env_addends.push_back(p_prime_ij_env.at(i,j)*current.w().at(i,j)); //restore divided factor to get z'
            }
        }
        //normalize P_{ij},P'^{env}_{ij}
        double sum_p_ij=vec_add_float(sum_p_ij_addends);
        double sum_p_prime_ij_env=vec_add_float(sum_p_prime_ij_env_addends);
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                // if(fabs(sum_p_ij)>1e-10){
                    p_ij.at(i,j)/=sum_p_ij;
                // }
                // if(fabs(sum_p_prime_ij_env)>1e-10){
                    p_prime_ij_env.at(i,j)/=sum_p_prime_ij_env;
                // }
            }
        }
        // std::cout<<(std::string)current.w()<<"\n";
        // std::cout<<(std::string)p_ij<<"\n";
        // std::cout<<(std::string)p_prime_ij_env<<"\n";

        std::vector<double> final_sum_ij_addends;
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                    if(fabs(p_prime_ij_env.at(i,j))>1e-10){
                        current.w().at(i,j)=p_ij.at(i,j)/p_prime_ij_env.at(i,j);
                        if(current.w().at(i,j)<1e-10){
                            current.w().at(i,j)=1e-10;
                        }
                    }
                    // final_sum_ij_addends.push_back(current.w().at(i,j));
                }
                else{ //lr!=0 means use gradient descent with lr
                    if(fabs(current.w().at(i,j))>1e-10){
                        // current.w().at(i,j)=p_ij.at(i,j)/p_prime_ij_env.at(i,j);
                        // current.w().at(i,j)-=lr*((p_prime_ij_env.at(i,j)*current.w().at(i,j))-p_ij.at(i,j))/current.w().at(i,j);
                        current.w().at(i,j)-=lr*(p_prime_ij_env.at(i,j)-(p_ij.at(i,j)/current.w().at(i,j)));
                        if(current.w().at(i,j)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                            current.w().at(i,j)=1e-10;
                        }
                    }
                    // final_sum_ij_addends.push_back(current.w().at(i,j));
                }
            }
        }
        // double final_sum_ij=vec_add_float(final_sum_ij_addends);
        // for(size_t i=0;i<current.w().nx();i++){
            // for(size_t j=0;j<current.w().ny();j++){
               // current.w().at(i,j)/=final_sum_ij;
            // }
        // }
        
        current.j(current.w().nx(),current.w()); //only valid for potts models with constant rank
        current.bmi(current.w());
        // std::cout<<"current.w():\n"<<(std::string)current.w()<<"\n";
        
        //DISABLE if using f_alg1, CAUSES INSTABILITY IN OPTIMIZATION
        // current.f()=optimize::f_maxent(sites[current.v1()],sites[current.v2()],current.w(),r_k);
        // std::cout<<(std::string) current.w()<<"\n";
        // std::cout<<(std::string) current.f()<<"\n";
        
        //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
        //p_prime_ki,imu if source==current.v1(), p_prime_kj,jnu if source==current.v2(). wlog, use p_prime_ki and imu.
        //k, the new index, is always second because virtual indices > physical indices.
        for(size_t n=0;n<cluster.size();n++){
            // std::cout<<"old cluster[n]: "<<(std::string) cluster[n].w()<<"\n";
            //determine whether this bond was connected to site i or to site j
            size_t source;
            if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
                source=(old_cluster[n].v1()==master)?master:slave;
            }
            else{
                source=(old_cluster[n].v2()==master)?master:slave;
            }
        
            /* std::stringstream ki_bmi_dump_line;
            std::stringstream ki_cost_dump_line; */
            array2d<double> p_ki(cluster[n].w().nx(),cluster[n].w().ny());
            array2d<double> p_prime_ki_env(cluster[n].w().nx(),cluster[n].w().ny());
            
            //prep to calculate P^_{ki_\mu}, P'^{env}_{ki_\mu}
            std::vector<std::vector<double> > addends(p_ki.nx()*p_ki.ny());
            std::vector<std::vector<double> > addends_env(p_prime_ki_env.nx()*p_prime_ki_env.ny());
            std::vector<double> sum_p_ki_addends;
            std::vector<double> sum_p_prime_ki_env_addends;
            cluster[n].bmi(cluster[n].w());
            std::vector<double> gi_prime(r_k,1);
            std::vector<double> gj_prime(r_k,1);
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    //calculate G'(f_k(S_i,S_j))
                    std::vector<double> gi_prime_factors;
                    std::vector<double> gj_prime_factors;
                    size_t k=current.f().at(i,j); //select where output is added to
                    for(size_t m=0;m<cluster.size();m++){
                        if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                            if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                gi_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                            }
                        }
                        else{ //connected to site j
                            if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                gj_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                            }
                        }
                    }
                    gi_prime[k]=vec_mult_float(gi_prime_factors);
                    gj_prime[k]=vec_mult_float(gj_prime_factors);
                }
            }
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    size_t k=current.f().at(i,j); //select where output is added to
                    //get addends
                    for(size_t imu=0;imu<p_ki.nx();imu++){
                        std::vector<double> factors;
                        std::vector<double> factors_env;
                        //calculate P_{ki_\mu} addends
                        factors.push_back(old_current.w().at(i,j));
                        if(source==current.v1()){ //connected to site i
                            if(old_cluster[n].v1()==current.v1()){ //site imu > site i
                                factors.push_back(old_cluster[n].w().at(i,imu));
                                factors.push_back(1/(old_cluster[n].w().sum_over_axis(1))[i]);
                            }
                            else{ //site imu < site i
                                factors.push_back(old_cluster[n].w().at(imu,i));
                                factors.push_back(1/(old_cluster[n].w().sum_over_axis(0))[i]);
                            }
                        }
                        else{ //connected to site j
                            if(old_cluster[n].v1()==current.v2()){ //site imu > site j
                                factors.push_back(old_cluster[n].w().at(j,imu));
                                factors.push_back(1/(old_cluster[n].w().sum_over_axis(1))[j]);
                            }
                            else{ //site imu < site j
                                factors.push_back(old_cluster[n].w().at(imu,j));
                                factors.push_back(1/(old_cluster[n].w().sum_over_axis(0))[j]);
                            }
                        }
                        factors.push_back(gi[i]);
                        factors.push_back(gj[j]);
                        addends[(k*p_ki.nx())+imu].push_back(vec_mult_float(factors));
                        sum_p_ki_addends.push_back(vec_mult_float(factors));
                        //calculate P'^{env}_{ki_\mu} addends
                        factors_env.push_back(current.w().at(i,j));
                        factors_env.push_back(gi_prime[k]);
                        factors_env.push_back(gj_prime[k]);
                        factors_env.push_back(1/(cluster[n].w().sum_over_axis(0))[k]); //divide appropriate factor
                        addends_env[(k*p_prime_ki_env.nx())+imu].push_back(vec_mult_float(factors_env));
                        sum_p_prime_ki_env_addends.push_back(vec_mult_float(factors_env)*cluster[n].w().at(imu,k)); //restore divided factor to get z'
                    }
                }
            }
            //calculate P^_{ki_\mu}, P'^{env}_{ki_\mu}
            for(size_t imu=0;imu<p_ki.nx();imu++){
                for(size_t k=0;k<p_ki.ny();k++){
                    p_ki.at(imu,k)=vec_add_float(addends[(k*p_ki.nx())+imu]);
                    p_prime_ki_env.at(imu,k)=vec_add_float(addends_env[(k*p_ki.nx())+imu]);
                }
            }
            double sum_p_ki=vec_add_float(sum_p_ki_addends);
            double sum_p_prime_ki_env=vec_add_float(sum_p_prime_ki_env_addends);
            // std::cout<<"p_ki: "<<(std::string) p_ki<<"\n";
            // std::cout<<"p_ki_env: "<<(std::string) p_prime_ki_env<<"\n";
            //normalize P_{ki_\mu},P'^{env}_{ki_\mu}
            for(size_t i=0;i<p_ki.nx();i++){
                for(size_t j=0;j<p_ki.ny();j++){
                    // if(fabs(sum_p_ki)>1e-10){
                        p_ki.at(i,j)/=sum_p_ki;
                    // }
                    // if(fabs(sum_p_prime_ki_env)>1e-10){
                        p_prime_ki_env.at(i,j)/=sum_p_prime_ki_env;
                    // }
                }
            }
            
            std::vector<double> final_sum_ki_addends;
            for(size_t i=0;i<p_ki.nx();i++){
                for(size_t j=0;j<p_ki.ny();j++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        if(fabs(p_prime_ki_env.at(i,j))>1e-10){
                            cluster[n].w().at(i,j)=p_ki.at(i,j)/p_prime_ki_env.at(i,j);
                            if(cluster[n].w().at(i,j)<1e-10){
                                cluster[n].w().at(i,j)=1e-10;
                            }
                        }
                        // final_sum_ki_addends.push_back(cluster[n].w().at(i,j));
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        if(fabs(cluster[n].w().at(i,j))>1e-10){
                            // cluster[n].w().at(i,j)=p_ki.at(i,j)/p_prime_ki_env.at(i,j);
                            // cluster[n].w().at(i,j)-=lr*((p_prime_ki_env.at(i,j)*cluster[n].w().at(i,j))-p_ki.at(i,j))/cluster[n].w().at(i,j);
                            cluster[n].w().at(i,j)-=lr*(p_prime_ki_env.at(i,j)-(p_ki.at(i,j)/cluster[n].w().at(i,j)));
                            if(cluster[n].w().at(i,j)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                cluster[n].w().at(i,j)=1e-10;
                            }
                        }
                        // final_sum_ki_addends.push_back(cluster[n].w().at(i,j));
                    }
                }
            }
            // std::cout<<"cluster[n]: "<<(std::string) cluster[n].w()<<"\n";
    
            // double final_sum_ki=vec_add_float(final_sum_ki_addends);
            // for(size_t i=0;i<cluster[n].w().nx();i++){
                // for(size_t j=0;j<cluster[n].w().ny();j++){
                    // cluster[n].w().at(i,j)/=final_sum_ki;
                // }
            // }
            cluster[n].j(cluster[n].w().nx(),cluster[n].w()); //only valid for potts models with constant rank
            cluster[n].bmi(cluster[n].w());
            
            /* //DEBUG OUTPUT
            ki_bmi_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cluster[n].bmi()<<"\n";
            // std::cout<<ki_eff_k_dump_lines.size()<<"\n";
            ki_bmi_dump_lines.push_back(ki_bmi_dump_line.str());
            //DEBUG: calculate cost function
            double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
            ki_cost_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
            ki_cost_dump_lines.push_back(ki_cost_dump_line.str()); */
        }
        /* //DEBUG OUTPUT
        ij_bmi_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<current.bmi()<<"\n";
        ij_bmi_dump_lines.push_back(ij_bmi_dump_line.str());
        //DEBUG: calculate cost function
        double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
        ij_cost_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
        ij_cost_dump_lines.push_back(ij_cost_dump_line.str()); */
    }
    /* p_prime_ij_opt_count_idx++;
    for(size_t i=0;i<ij_bmi_dump_lines.size();i++){
        ij_ofs<<ij_bmi_dump_lines[i];
        ij_cost_ofs<<ij_cost_dump_lines[i];
    }
    p_prime_ki_opt_count_idx+=cluster.size();
    for(size_t i=0;i<ki_bmi_dump_lines.size();i++){
        ki_ofs<<ki_bmi_dump_lines[i];
        ki_cost_ofs<<ki_cost_dump_lines[i];
    } */
    // std::cout<<(std::string) current.w()<<"\n";
}

//cmd function mimicking alg. 1. this assumes and results in a fixed rank.
array2d<size_t> optimize::f_alg1(site v_i,site v_j){
    array2d<size_t> f_res(v_i.rank(),v_j.rank());
    for(size_t s_i=0;s_i<v_i.rank();s_i++){
        for(size_t s_j=0;s_j<v_j.rank();s_j++){
            // f_res.at(s_i,s_j)=(v_i.vol()>=v_j.vol())?s_i:s_j;
            f_res.at(s_i,s_j)=s_i;
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
    // std::cout<<(std::string)current<<"\n";
    // std::cout<<(std::string)current.f()<<"\n";
    // std::cout<<(std::string)current.w()<<"\n";
    
    std::vector<double> g_prime(*(std::max_element(current.f().e().begin(),current.f().e().end()))+1);
    for(size_t k=0;k<g_prime.size();k++){
        double g_prime_res=1;
        for(size_t n=0;n<cluster.size();n++){
            double spin_sum_prime=0;
            for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                spin_sum_prime+=cluster[n].w().at(imu,k);
            }
            g_prime_res*=spin_sum_prime;
        }
        g_prime[k]=g_prime_res;
    }
    
    std::vector<double> gi(sites[old_current.v1()].rank());
    std::vector<double> gj(sites[old_current.v2()].rank());
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        double gi_res=1;
        for(size_t n=0;n<old_cluster.size();n++){
            double spin_sum_i=0;
            size_t source;
            if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
            }
            else{
                source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
            }
            if(source==old_current.v1()){ //connected to site i
                if(old_cluster[n].v1()==old_current.v2()){ //site imu > site i
                    for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                        spin_sum_i+=old_cluster[n].w().at(i,imu);
                    }
                }
                else{ //site imu < site i
                    for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                        spin_sum_i+=old_cluster[n].w().at(imu,i);
                    }
                }
                gi_res*=spin_sum_i;
            }
        }
        gi[i]=gi_res;
    }
    for(size_t j=0;j<sites[current.v2()].rank();j++){
        double gj_res=1;
        for(size_t n=0;n<old_cluster.size();n++){
            double spin_sum_j=0;
            size_t source;
            if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
            }
            else{
                source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
            }
            if(source==old_current.v2()){ //connected to site j
                if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                    for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                        spin_sum_j+=old_cluster[n].w().at(j,imu);
                    }
                }
                else{ //site imu < site j
                    for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                        spin_sum_j+=old_cluster[n].w().at(imu,j);
                    }
                }
                gj_res*=spin_sum_j;
            }
        }
        gj[j]=gj_res;
    }
    
    array2d<double> gi_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    array2d<double> gj_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double gi_tilde_res=0;
            double gj_tilde_res=0;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum_i_tilde=0;
                double spin_sum_j_tilde=0;
                double div_factor=0;
                size_t k=current.f().at(i,j);
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
                            div_factor+=old_cluster[n].w().at(i,imu);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_i_tilde+=old_cluster[n].w().at(i,imu)*log(cluster[n].w().at(imu,k));
                            }
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=old_cluster[n].w().at(imu,i);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_i_tilde+=old_cluster[n].w().at(imu,i)*log(cluster[n].w().at(imu,k));
                            }
                        }
                    }
                    gi_tilde_res+=(div_factor>0)?(spin_sum_i_tilde*gi[i]/div_factor):0;
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=old_cluster[n].w().at(j,imu);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_j_tilde+=old_cluster[n].w().at(j,imu)*log(cluster[n].w().at(imu,k));
                            }
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=old_cluster[n].w().at(imu,j);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_j_tilde+=old_cluster[n].w().at(imu,j)*log(cluster[n].w().at(imu,k));
                            }
                        }
                    }
                    gj_tilde_res+=(div_factor>0)?(spin_sum_j_tilde*gj[j]/div_factor):0;
                }
            }
            gi_tilde.at(i,j)=gi_tilde_res;
            gj_tilde.at(i,j)=gj_tilde_res;
        }
    }
    
    double z_prime=0;
    double z=0;
    double a=0;
    double b=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib_z_prime=current.w().at(i,j)*g_prime[current.f().at(i,j)];
            double contrib_z=old_current.w().at(i,j)*gi[i]*gj[j];
            double contrib_a=(current.w().at(i,j)>0)?old_current.w().at(i,j)*gi[i]*gj[j]*log(current.w().at(i,j)):0;
            double contrib_b=old_current.w().at(i,j)*((gi_tilde.at(i,j)*gj[j])+(gi[i]*gj_tilde.at(i,j)));
            size_t k=current.f().at(i,j);
            double contrib_b_logs=0;
            z_prime+=contrib_z_prime;
            z+=contrib_z;
            a+=contrib_a;
            b+=contrib_b;
        }
    }
    // std::cout<<"z: "<<z<<"\nz': "<<z_prime<<"\nz': "<<z_prime2<<"\n";
    // std::cout<<"a: "<<a<<"\nb: "<<b<<"\n";
    double res=log(z_prime)-(a/z)-(b/z);
    return res;
}