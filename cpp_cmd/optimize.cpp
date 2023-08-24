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
    //precompute intermediate quantities for weight optimization
    std::vector<double> orig_g1(current.w().nx(),1);
    std::vector<double> orig_g2(current.w().ny(),1);
    std::vector<double> g1(current.w().nx(),1);
    std::vector<double> g2(current.w().ny(),1);
    array2d<double> pair_ij(current.w().nx(),current.w().ny());
    array2d<double> pair_ij_env(current.w().nx(),current.w().ny());
    
    for(size_t i=0;i<current.w().nx();i++){
        // orig_g1[i]=1;
        std::vector<double> orig_g1_factors;
        for(auto it=sites[old_current.v1()].adj().begin();it!=sites[old_current.v1()].adj().end();++it){
            // std::cout<<"g1_contrib: "<<((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i]<<"\n";
            // orig_g1[i]*=((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i];
            // orig_g1_factors.push_back(((*it).w().sum_over_axis(((*it).v1()==current.v1())?0:1))[i]);
            orig_g1_factors.push_back(((*it).w().sum_over_axis(((*it).v1()==old_current.v1())?1:0))[i]);
        }
        orig_g1[i]=vec_mult_float(orig_g1_factors);
    }
    for(size_t j=0;j<current.w().ny();j++){
        // orig_g2[j]=1;
        std::vector<double> orig_g2_factors;
        for(auto it=sites[old_current.v2()].adj().begin();it!=sites[old_current.v2()].adj().end();++it){
            // std::cout<<"g2_contrib: "<<((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j]<<"\n";
            // orig_g2[j]*=((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j];
            // orig_g2_factors.push_back(((*it).w().sum_over_axis(((*it).v1()==current.v2())?0:1))[j]);
            orig_g2_factors.push_back(((*it).w().sum_over_axis(((*it).v1()==old_current.v2())?1:0))[j]);
        }
        orig_g2[j]=vec_mult_float(orig_g2_factors);
    }
    
    std::vector<std::string> ij_bmi_dump_lines;
    std::vector<std::string> ij_cost_dump_lines;
    std::vector<std::string> ki_bmi_dump_lines;
    std::vector<std::string> ki_cost_dump_lines;
    // std::cout<<"IJ opt for bond between sites ("<<current.v1()<<","<<current.v2()<<")\n";
    double sum_pair_ij=0;
    std::vector<double> sum_pair_ij_addends;
    for(size_t i=0;i<pair_ij.nx();i++){
        for(size_t j=0;j<pair_ij.ny();j++){
            pair_ij.at(i,j)=current.w().at(i,j)*orig_g1[i]*orig_g2[j];
            // sum_pair_ij+=pair_ij.at(i,j);
            sum_pair_ij_addends.push_back(pair_ij.at(i,j));
        }
    }
    sum_pair_ij=vec_add_float(sum_pair_ij_addends);
        
    // std::cout<<(std::string) current.f()<<"\n";
    
    for(size_t n1=0;n1<max_it;n1++){
        std::stringstream ij_bmi_dump_line;
        std::stringstream ij_cost_dump_line;
        //calculate pair_{ij}
        //TODO: calculate pair'_{ij}^{env} (uses cm func), can be reused for pair'_{ki_mu}^{env}?
        double sum_pair_ij_env=0;
        std::vector<double> sum_pair_ij_env_addends;
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                // double g1_env=1;
                // double g2_env=1;
                std::vector<double> g1_env_factors;
                std::vector<double> g2_env_factors;
                size_t k=current.f().at(i,j); //select where output is added to
                for(auto it=sites[current.v1()].adj().begin();it!=sites[current.v1()].adj().end();++it){
                    // g1_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                    g1_env_factors.push_back(((*it).w().sum_over_axis(0))[k]);
                }
                for(auto it=sites[current.v2()].adj().begin();it!=sites[current.v2()].adj().end();++it){
                    // g2_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                    g2_env_factors.push_back(((*it).w().sum_over_axis(0))[k]);
                }
                double g1_env=vec_mult_float(g1_env_factors);
                double g2_env=vec_mult_float(g2_env_factors);
                pair_ij_env.at(i,j)=g1_env*g2_env;
                // sum_pair_ij_env+=current.w().at(i,j)*pair_ij_env.at(i,j);
                sum_pair_ij_env_addends.push_back(current.w().at(i,j)*pair_ij_env.at(i,j));
            }
        }
        sum_pair_ij_env=vec_add_float(sum_pair_ij_env_addends);
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                pair_ij.at(i,j)/=sum_pair_ij;
                pair_ij_env.at(i,j)/=sum_pair_ij_env;
            }
        }
        for(size_t i=0;i<pair_ij.nx();i++){
            for(size_t j=0;j<pair_ij.ny();j++){
                if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                    current.w().at(i,j)=pair_ij.at(i,j)/pair_ij_env.at(i,j);
                }
                else{ //lr!=0 means use gradient descent with lr
                    current.w().at(i,j)-=lr*((pair_ij_env.at(i,j)*current.w().at(i,j))-pair_ij.at(i,j))/current.w().at(i,j);
                }
            }
        }
        // std::cout<<"prefinal current.w():\n"<<(std::string)current.w()<<"\n";
        current.j(current.w().nx(),current.w()); //only valid for potts models with constant rank
        current.bmi(current.w());
        // std::cout<<"final current.w():\n"<<(std::string)current.w()<<"\n";
        
        //DISABLE if using f_alg1
        current.f()=optimize::f_maxent(sites[current.v1()],sites[current.v2()],current.w(),r_k);
        
        //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
        //pair_ki,imu if source==current.v1(), pair_kj,jnu if source==current.v2(). wlog, use pair_ki and imu.
        //k, the new index, is always second because virtual indices > physical indices.
        //TODO: fix for variable r_k...
        // std::cout<<"HERE kl_iterative\n";
        for(size_t n=0;n<cluster.size();n++){
            size_t source;
            if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
                source=(old_cluster[n].v1()==master)?master:slave;
            }
            else{
                source=(old_cluster[n].v2()==master)?master:slave;
            }
            // std::cout<<source<<" "<<current.v1()<<" "<<current.v2()<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<"\n";
            // std::cout<<"KI opt for bond between sites ("<<cluster[n].v1()<<","<<cluster[n].v2()<<")\n";
        
            // std::cout<<"HERE opt\n";
            std::stringstream ki_bmi_dump_line;
            std::stringstream ki_cost_dump_line;
            array2d<double> pair_ki(cluster[n].w().nx(),cluster[n].w().ny());
            array2d<double> pair_ki_env(cluster[n].w().nx(),cluster[n].w().ny());
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
            //TODO: if rank increased, reinitialize cluster[n].w() with new correct size.
            // std::cout<<"HERE reinit rank\n";
            if((pair_ki.nx()!=cluster[n].w().nx())||(pair_ki.ny()!=cluster[n].w().ny())){
                // std::cout<<(std::string)cluster[n].w()<<"\n";
                array2d<double> new_w(pair_ki.nx(),pair_ki.ny());
                // std::cout<<sum_pair_ki<<"\n";
                // std::cout<<sum_pair_ki_env<<"\n";
                // std::cout<<(std::string)pair_ki<<"\n";
                // std::cout<<(std::string)pair_ki_env<<"\n";
                // for(size_t i=0;i<cluster[n].w().nx();i++){
                    // for(size_t j=0;j<cluster[n].w().ny();j++){
                        // new_w.at(i,j)=cluster[n].w().at(i,j);
                    // }
                // }
                for(size_t i=0;i<new_w.nx();i++){
                    for(size_t j=0;j<new_w.ny();j++){
                        if((i<cluster[n].w().nx())&&(j<cluster[n].w().ny())){
                            new_w.at(i,j)=cluster[n].w().at(i,j)*(cluster[n].w().nx()*cluster[n].w().ny())/(double) (new_w.nx()*new_w.ny());
                        }
                        else{
                            new_w.at(i,j)=1/(double) (new_w.nx()*new_w.ny());
                        }
                    }
                }
                cluster[n].w()=new_w;
                // std::cout<<(std::string)cluster[n].w()<<"\n";
            }
            // std::cout<<"HERE reinit rank done\n";
            double sum_pair_ki=0;
            double sum_pair_ki_env=0;
            std::vector<std::vector<double> > addends(pair_ki.nx()*pair_ki.ny());
            std::vector<std::vector<double> > addends_env(pair_ki_env.nx()*pair_ki_env.ny());
            std::vector<double> sum_pair_ki_addends;
            std::vector<double> sum_pair_ki_env_addends;
            // std::cout<<"HERE imu opt\n";
            // std::cout<<(std::string)current.w()<<"\n";
            // std::cout<<(std::string)current.f()<<"\n";
            // std::cout<<(std::string)old_cluster[n].w()<<"\n";
            // std::cout<<(std::string)cluster[n].w()<<"\n";
            cluster[n].bmi(cluster[n].w());
            // cluster[n].f()=optimize::f_maxent(sites[cluster[n].v1()],sites[cluster[n].v2()],cluster[n].w(),r_k);
            // for(size_t i=0;i<cluster[n].w().nx();i++){
                // for(size_t j=0;j<cluster[n].w().ny();j++){
                    // size_t k=cluster[n].f().at(i,j); //select where output is added to
            // std::cout<<"source: "<<source<<"\n";
            // std::cout<<"current v: "<<current.v1()<<" "<<current.v2()<<"\n";
            // std::cout<<"old_cluster[n] v: "<<old_cluster[n].v1()<<" "<<old_cluster[n].v2()<<"\n";
            // std::cout<<"cluster[n] v: "<<cluster[n].v1()<<" "<<cluster[n].v2()<<"\n";
            // std::cout<<"current w size: "<<current.w().nx()<<" "<<current.w().ny()<<"\n";
            // std::cout<<"old_cluster[n] w size: "<<old_cluster[n].w().nx()<<" "<<old_cluster[n].w().ny()<<"\n";
            // std::cout<<"cluster[n] w size: "<<cluster[n].w().nx()<<" "<<cluster[n].w().ny()<<"\n";
            // std::cout<<"pair_ki w size: "<<pair_ki.nx()<<" "<<pair_ki.ny()<<"\n";
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    size_t k=current.f().at(i,j); //select where output is added to
                    double orig_g1_ki=orig_g1[i];
                    double orig_g2_ki=orig_g2[j];
                    // double g1_ki_env=g1[i]; //??? what was g1 again?
                    // double g2_ki_env=g2[j];
                    std::vector<double> g1_ki_env_factors;
                    std::vector<double> g2_ki_env_factors;
                    for(auto it=sites[current.v1()].adj().begin();it!=sites[current.v1()].adj().end();++it){
                        // g1_ki_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                        g1_ki_env_factors.push_back(((*it).w().sum_over_axis(0))[k]);
                    }
                    for(auto it=sites[current.v2()].adj().begin();it!=sites[current.v2()].adj().end();++it){
                        // g2_ki_env*=((*it).w().sum_over_axis(0))[k]; //new site is always v2 so sum over axis 0
                        g2_ki_env_factors.push_back(((*it).w().sum_over_axis(0))[k]);
                    }
                    double g1_ki_env=vec_mult_float(g1_ki_env_factors);
                    double g2_ki_env=vec_mult_float(g2_ki_env_factors);
                    
                    if(source==current.v1()){
                        // std::cout<<"source=current.v1()\n";
                        bool flag=(old_cluster[n].v1()==old_current.v1());
                        bool flag_env=(cluster[n].v1()==current.v1());
                        // std::cout<<"flags: "<<flag<<" "<<flag_env<<"\n";
                        orig_g1_ki/=(old_cluster[n].w().sum_over_axis(flag?1:0))[i];
                        // g1_ki_env/=(cluster[n].w().sum_over_axis(flag_env?1:0))[i]; //always ax0 because ax1 always refers to new virtual site
                        g1_ki_env/=(cluster[n].w().sum_over_axis(0))[k]; //always ax0 because ax1 always refers to new virtual site
                    }
                    else{
                        // std::cout<<"source=current.v2()\n";
                        bool flag=(old_cluster[n].v1()==old_current.v2());
                        bool flag_env=(cluster[n].v1()==current.v2());
                        // std::cout<<"flags: "<<flag<<" "<<flag_env<<"\n";
                        orig_g2_ki/=(old_cluster[n].w().sum_over_axis(flag?1:0))[j];
                        // g2_ki_env/=(cluster[n].w().sum_over_axis(flag_env?1:0))[j]; //always ax0 because ax1 always refers to new virtual site
                        g2_ki_env/=(cluster[n].w().sum_over_axis(0))[k]; //always ax0 because ax1 always refers to new virtual site
                    }
                    // std::cout<<orig_g1_ki<<" "<<g1_ki_env<<"\n";
                    // std::cout<<orig_g2_ki<<" "<<g2_ki_env<<"\n";
                    // std::cout<<"g1*g2: "<<(g1_ki*g2_ki)<<"\n";
                    // std::cout<<std::string(current.w());
                    // std::cout<<"HERE imu add\n";
                    for(size_t imu=0;imu<pair_ki.nx();imu++){
                        // std::cout<<r_k<<" "<<i<<" "<<j<<" "<<k<<" "<<imu<<"\n";
                        //TODO: how to reinitialize cluster[n].w() for r_max>q? current iteration appears to worsen the approximation...
                        // std::cout<<imu<<" "<<k<<" "<<pair_ki.nx()<<" "<<addends.size()<<" "<<((k*pair_ki.nx())+imu)<<"\n";
                        // std::cout<<imu<<" "<<k<<" "<<pair_ki_env.nx()<<" "<<addends_env.size()<<" "<<((k*pair_ki_env.nx())+imu)<<"\n";
                        std::vector<double> factors;
                        std::vector<double> factors_env;
                        factors.push_back(old_current.w().at(i,j));
                        factors.push_back(orig_g1_ki);
                        factors.push_back(orig_g2_ki);
                        // sum_pair_ki+=vec_mult_float(factors);
                        sum_pair_ki_addends.push_back(vec_mult_float(factors));
                        // factors.push_back(old_cluster[n].w().at((old_cluster[n].v1()==old_current.v1())?i:imu,(old_cluster[n].v1()==old_current.v1())?imu:j)); //pick from w_{ii_mu} or w_{i_muj}
                        // factors.push_back(old_cluster[n].w().at((old_cluster[n].v1()==old_current.v1())?imu:j,(old_cluster[n].v1()==old_current.v1())?i:imu)); //pick from w_{ii_mu} or w_{i_muj}
                        // factors.push_back(old_cluster[n].w().at((old_cluster[n].v1()==current.v1())?imu:j,(old_cluster[n].v1()==current.v1())?i:imu));
                        // factors.push_back(old_cluster[n].w().at(imu,(source==current.v1())?i:j));
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
                        factors_env.push_back(current.w().at(i,j));
                        factors_env.push_back(g1_ki_env);
                        factors_env.push_back(g2_ki_env);
                        // std::cout<<"prod: "<<vec_mult_float(factors)<<"\n";
                        // std::cout<<"prod (env): "<<vec_mult_float(factors_env)<<"\n";
                        addends[(k*pair_ki.nx())+imu].push_back(vec_mult_float(factors));
                        addends_env[(k*pair_ki_env.nx())+imu].push_back(vec_mult_float(factors_env));
                        // sum_pair_ki_env+=vec_mult_float(factors_env);
                        sum_pair_ki_env_addends.push_back(vec_mult_float(factors_env));
                        
                        // addends[(k*pair_ki.nx())+imu].push_back(old_current.w().at(i,j)*old_cluster[n].w().at(imu,(source==current.v1())?i:j)*orig_g1_ki*orig_g2_ki);
                        // addends_env[(k*pair_ki_env.nx())+imu].push_back(current.w().at(i,j)*g1_ki_env*g2_ki_env);
                        // sum_pair_ki+=old_current.w().at(i,j)*orig_g1_ki*orig_g2_ki;
                        // sum_pair_ki_env+=current.w().at(i,j)*g1_ki_env*g2_ki_env;
                    }
                    // std::cout<<"HERE imu add done\n";
                }
            }
            // std::cout<<sum_pair_ki<<"\n";
            // std::cout<<sum_pair_ki_env<<"\n";
            // std::cout<<"HERE imu opt done\n";
            // std::cout<<(std::string)pair_ki<<"\n";
            for(size_t imu=0;imu<pair_ki.nx();imu++){
                for(size_t k=0;k<pair_ki.ny();k++){
                    pair_ki.at(imu,k)=vec_add_float(addends[(k*pair_ki.nx())+imu]);
                    pair_ki_env.at(imu,k)=vec_add_float(addends_env[(k*pair_ki.nx())+imu]);
                }
            }
            // std::cout<<(std::string)pair_ki<<"\n";
            sum_pair_ki=vec_add_float(sum_pair_ki_addends);
            sum_pair_ki_env=vec_add_float(sum_pair_ki_env_addends);
            for(size_t i=0;i<pair_ki.nx();i++){
                for(size_t j=0;j<pair_ki.ny();j++){
                    pair_ki.at(i,j)/=sum_pair_ki;
                    pair_ki_env.at(i,j)/=sum_pair_ki_env;
                }
            }
            // std::cout<<"pre-reinit pair_ki:\n"<<(std::string)pair_ki<<"\n";
            // std::cout<<"pre-reinit pair_ki_env:\n"<<(std::string)pair_ki_env<<"\n";
            
            // std::cout<<sites[current.v1()].rank()<<" "<<sites[current.v2()].rank()<<" "<<sites[cluster[n].v1()].rank()<<" "<<sites[cluster[n].v2()].rank()<<"\n";
            // std::cout<<current.v1()<<" "<<current.v2()<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<"\n";
            
            for(size_t i=0;i<pair_ki.nx();i++){
                for(size_t j=0;j<pair_ki.ny();j++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        if(pair_ki_env.at(i,j)!=0){
                            cluster[n].w().at(i,j)=pair_ki.at(i,j)/pair_ki_env.at(i,j);
                        }
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        cluster[n].w().at(i,j)-=lr*((pair_ki_env.at(i,j)*cluster[n].w().at(i,j))-pair_ki.at(i,j))/cluster[n].w().at(i,j);
                    }
                }
            }
            // std::cout<<"prefinal cluster[n].w():\n"<<(std::string)cluster[n].w()<<"\n";
            cluster[n].j(cluster[n].w().nx(),cluster[n].w()); //only valid for potts models with constant rank
            cluster[n].bmi(cluster[n].w());
            
            //DEBUG OUTPUT
            if(n1>0){
                ki_bmi_dump_line<<(pair_opt_ki_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cluster[n].bmi()<<"\n";
                // std::cout<<ki_eff_k_dump_lines.size()<<"\n";
                ki_bmi_dump_lines.push_back(ki_bmi_dump_line.str());
                //DEBUG: calculate cost function
                double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
                ki_cost_dump_line<<(pair_opt_ki_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
                ki_cost_dump_lines.push_back(ki_cost_dump_line.str());
            }
            
            // std::cout<<"HERE opt done\n";
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
    double res=0;
    double z=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib=old_current.w().at(i,j);
            for(size_t n=0;n<cluster.size();n++){
                // size_t site_idx=((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].v2():old_cluster[n].v1();
                // double spin_sum=0;
                // for(size_t imu=0;imu<sites[site_idx].rank();imu++){
                    // spin_sum+=(site_idx==old_cluster[n].v2())?old_cluster[n].w().at(i,imu):old_cluster[n].w().at(imu,i);
                // }
                // contrib*=spin_sum;
                
                size_t source,neighbor;
                double spin_sum=0;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                    source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
                }
                else{
                    source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
                }
                neighbor=(old_cluster[n].v1()==source)?old_cluster[n].v2():old_cluster[n].v1();
                // for(size_t imu=0;imu<sites[neighbor].rank();imu++){
                    // if(source==old_current.v1()){ //connected to site i
                        // if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                            // spin_sum+=old_cluster[n].w().at(i,imu);
                        // }
                        // else{ //site imu < site i
                            // spin_sum+=old_cluster[n].w().at(imu,i);
                        // }
                    // }
                    // else{ //connected to site j
                        // if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                            // spin_sum+=old_cluster[n].w().at(j,imu);
                        // }
                        // else{ //site imu < site j
                            // spin_sum+=old_cluster[n].w().at(imu,j);
                        // }
                    // }
                // }
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
                for(size_t imu=0;imu<cluster[n].w().nx();imu++){
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
                // size_t site_idx=((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].v2():old_cluster[n].v1();
                // double spin_sum=0;
                // for(size_t imu=0;imu<sites[site_idx].rank();imu++){
                    // spin_sum+=(site_idx==old_cluster[n].v2())?old_cluster[n].w().at(i,imu):old_cluster[n].w().at(imu,i);
                // }
                // contrib*=spin_sum;
                
                size_t source,neighbor;
                double spin_sum=0;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                    source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
                }
                else{
                    source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
                }
                neighbor=(old_cluster[n].v1()==source)?old_cluster[n].v2():old_cluster[n].v1();
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
                // size_t site_idx=((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].v2():old_cluster[n].v1();
                // double spin_sum=0;
                // for(size_t imu=0;imu<sites[site_idx].rank();imu++){
                    // spin_sum+=(site_idx==old_cluster[n].v2())?old_cluster[n].w().at(i,imu)*log(current.w().at(imu,current.f().at(i,j))):old_cluster[n].w().at(imu,i)*log(current.w().at(imu,current.f().at(i,j)));
                // }
                // contrib*=spin_sum;
                
                size_t source,neighbor;
                double spin_sum=0;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                    source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
                }
                else{
                    source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
                }
                neighbor=(old_cluster[n].v1()==source)?old_cluster[n].v2():old_cluster[n].v1();
                // std::cout<<old_current.v1()<<" "<<old_current.v2()<<" "<<old_cluster[n].v1()<<" "<<old_cluster[n].v2()<<"\n";
                // std::cout<<current.v1()<<" "<<current.v2()<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<"\n";
                // std::cout<<source<<" "<<neighbor<<"\n";
                // std::cout<<(std::string) old_cluster[n].w()<<"\n";
                // std::cout<<(std::string) cluster[n].w()<<"\n";
                // std::cout<<(std::string) current.f()<<"\n";
                if(source==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            // std::cout<<i<<","<<j<<","<<imu<<" "<<old_cluster[n].w().at(i,imu)<<" "<<cluster[n].w().at(imu,current.f().at(i,j))<<" "<<log(cluster[n].w().at(imu,current.f().at(i,j)))<<"\n";
                            spin_sum+=old_cluster[n].w().at(i,imu)*log(cluster[n].w().at(imu,current.f().at(i,j)));
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            // std::cout<<i<<","<<j<<","<<imu<<" "<<old_cluster[n].w().at(imu,i)<<" "<<cluster[n].w().at(imu,current.f().at(i,j))<<" "<<log(cluster[n].w().at(imu,current.f().at(i,j)))<<"\n";
                            spin_sum+=old_cluster[n].w().at(imu,i)*log(cluster[n].w().at(imu,current.f().at(i,j)));
                        }
                    }
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            // std::cout<<i<<","<<j<<","<<imu<<" "<<old_cluster[n].w().at(j,imu)<<" "<<cluster[n].w().at(imu,current.f().at(i,j))<<" "<<log(cluster[n].w().at(imu,current.f().at(i,j)))<<"\n";
                            spin_sum+=old_cluster[n].w().at(j,imu)*log(cluster[n].w().at(imu,current.f().at(i,j)));
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            // std::cout<<i<<","<<j<<","<<imu<<" "<<old_cluster[n].w().at(imu,j)<<" "<<cluster[n].w().at(imu,current.f().at(i,j))<<" "<<log(cluster[n].w().at(imu,current.f().at(i,j)))<<"\n";
                            spin_sum+=old_cluster[n].w().at(imu,j)*log(cluster[n].w().at(imu,current.f().at(i,j)));
                        }
                    }
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
 