#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <getopt.h>

#include "mpi.h"
#include "sampling.hpp"
#include "stopwatch.hpp"
#include "mpi_utils.hpp"
#include "graph.hpp"
#include "graph_utils.hpp"
#include "algorithm_nll.hpp"
#include "observables.hpp"

void print_usage(){
    std::cerr<<"usage: tree_ml_approx [--options] <q> <n_samples> <d> <{l|l0,l1,...}> <min_beta> <max_beta> <step_beta>\n";
    std::cerr<<"if <n_samples>!=0, required: -d, -1, -2, forbidden: -i,\n";
    std::cerr<<"options:\n";
    std::cerr<<"\t--open-bc: generate graph connectivity with open boundary conditions.\n";
    std::cerr<<"\t--use-t: min_beta, max_beta, and step_beta refer to temperature instead.\n";
    std::cerr<<"\t--rand-mc: MC is done with random initial state for the Markov chain instead of using the tree approximation.\n";
    std::cerr<<"\t--no-ti: force disable translational invariance.\n";
    std::cerr<<"\t-h,--help: display this message\n";
    std::cerr<<"\t-v,--verbose:\n\t\t0->nothing printed to stdout (forced for MPI)\n\t\t1->sample number and aggregate timing data\n\t\t2->per-instance timing\n\t\t3->more detailed timing breakdown\n\t\t4->graph contents, debug observable data\n";
    std::cerr<<"\t-o,--output: prefix for output files. please omit the file extension.\n";
    std::cerr<<"\t-d,--distribution: distribution for sampling bond configurations. one of \"gaussian\",\"bimodal\" (+1/-1),\"uniform\".\n";
    std::cerr<<"\t-1,--dist-param1: distribution hyperparameter.\n\t\tif gaussian-> mean\n\t\tif bimodal -> probability of ferromagnetic bond\n\t\tif uniform -> minimum bond strength\n";
    std::cerr<<"\t-2,--dist-param2: distribution hyperparameter.\n\t\tif gaussian-> standard deviation\n\t\tif bimodal -> ignored, overriden to 0\n\t\tif uniform -> maximum bond strength\n";
    std::cerr<<"\t-T,--init-tree-type: initial tree type: mps or pbttn\n";
    std::cerr<<"\t-r,--r-max: maximum rank of spins in the approximation\n";
    std::cerr<<"\t-s,--samples: number of samples to obtain per temperature\n";
    std::cerr<<"\t-c,--cycles: number of NLL training cycles per temperature\n";
    std::cerr<<"\t-N,--nll-iter-max: maximum number of NLL optimization iterations\n";
    std::cerr<<"\t-w,--min-n-sweeps: minimum number of sweeps in local MH update before drawing sample\n";
    std::cerr<<"\t-W,--max-n-sweeps: maximum number of sweeps in local MH update before drawing sample\n";
    std::cerr<<"\t-D,--step-n-sweeps: step in n_sweeps\n";
}

template<typename cmp>
graph<cmp> gen_lattice(size_t q,size_t r_max,std::vector<size_t> ls,bool open_bc,std::string dist_type,double dist_param1,double dist_param2,double beta,std::string init_tree_type){ //transformations are done to counteract the transformations in gen_hypercubic
    graph<cmp> g;
    if(dist_type=="gaussian"){
        //dist_param1=mean, dist_param2=std
        std::normal_distribution<double> dist((dist_param1+1)/2.0,dist_param2/2.0);
        if(init_tree_type=="pbttn"){
            g=graph_utils::init_pbttn<std::normal_distribution<double>,cmp>(q,r_max,ls,!open_bc,dist,beta);
        }
        else if(init_tree_type=="mps"){
            g=graph_utils::init_mps<std::normal_distribution<double>,cmp>(q,r_max,ls,!open_bc,dist,beta);
        }
    }
    else if(dist_type=="bimodal"){
        //dist_param1=p, dist_param2=N/A
        std::discrete_distribution<int> dist{1-dist_param1,dist_param1};
        if(init_tree_type=="pbttn"){
            g=graph_utils::init_pbttn<std::discrete_distribution<int>,cmp>(q,r_max,ls,!open_bc,dist,beta);
        }
        else if(init_tree_type=="mps"){
            g=graph_utils::init_mps<std::discrete_distribution<int>,cmp>(q,r_max,ls,!open_bc,dist,beta);
        }
    }
    else if(dist_type=="uniform"){
        //dist_param1=min, dist_param2=max
        std::uniform_real_distribution<double> dist{(dist_param1+1)/2.0,(dist_param2+1)/2.0};
        if(init_tree_type=="pbttn"){
            g=graph_utils::init_pbttn<std::uniform_real_distribution<double>,cmp>(q,r_max,ls,!open_bc,dist,beta);
        }
        else if(init_tree_type=="mps"){
            g=graph_utils::init_mps<std::uniform_real_distribution<double>,cmp>(q,r_max,ls,!open_bc,dist,beta);
        }
    }
    return g;
}

int main(int argc,char **argv){
    //mpi init
    mpi_utils::init();
    //argument handling
    int open_bc=0;
    int use_t=0;
    int rand_mc=0;
    int no_ti=0;
    size_t verbose=0;
    bool output_set=false;
    bool dist_set=false;
    bool dist_param1_set=false;
    bool dist_param2_set=false;
    std::string input,output,dist;
    std::string init_tree_type="pbttn";
    double dist_param1,dist_param2;
    size_t r_max=0;
    size_t n_config_samples=1000;
    size_t n_cycles=0;
    size_t n_nll_iter_max=10000;
    size_t min_n_sweeps=100;
    size_t max_n_sweeps=100;
    size_t step_n_sweeps=10;
    //option arguments
    while(1){
        static struct option long_opts[]={
            {"open-bc",no_argument,&open_bc,1},
            {"use-t",no_argument,&use_t,1},
            {"rand-mc",no_argument,&rand_mc,1},
            {"no-ti",no_argument,&no_ti,1},
            {"help",no_argument,0,'h'},
            {"verbose",required_argument,0,'v'},
            {"output",required_argument,0,'o'},
            {"distribution",required_argument,0,'d'},
            {"dist-param1",required_argument,0,'1'},
            {"dist-param2",required_argument,0,'2'},
            {"init-tree-type",required_argument,0,'T'},
            {"r-max",required_argument,0,'r'},
            {"samples",required_argument,0,'s'},
            {"cycles",required_argument,0,'c'},
            {"nll-iter-max",required_argument,0,'N'},
            {"min-n-sweeps",required_argument,0,'w'},
            {"max-n-sweeps",required_argument,0,'W'},
            {"step-n-sweeps",required_argument,0,'D'},
            {0, 0, 0, 0}
        };
        int opt_idx=0;
        int c=getopt_long(argc,argv,"hv:o:d:1:2:T:r:s:c:N:w:W:D:",long_opts,&opt_idx);
        if(c==-1){break;} //end of options
        switch(c){
            //handle long option flags
            case 0:
            break;
            //handle standard options
            case 'h': print_usage(); exit(1);
            case 'v': verbose=(size_t) atoi(optarg); break;
            case 'o': output=std::string(optarg); output_set=true; break;
            case 'd': dist=std::string(optarg); dist_set=true; break;
            case '1': dist_param1=(double) atof(optarg); dist_param1_set=true; break;
            case '2': dist_param2=(double) atof(optarg); dist_param2_set=true; break;
            case 'T': init_tree_type=std::string(optarg); break;
            case 'r': r_max=(size_t) atoi(optarg); break;
            case 's': n_config_samples=(size_t) atoi(optarg); break;
            case 'c': n_cycles=(size_t) atoi(optarg); break;
            case 'N': n_nll_iter_max=(size_t) atoi(optarg); break;
            case 'w': min_n_sweeps=(size_t) atoi(optarg); break;
            case 'W': max_n_sweeps=(size_t) atoi(optarg); break;
            case 'D': step_n_sweeps=(size_t) atoi(optarg); break;
            case '?':
            //error printed
            exit(1);
            default:
            if(mpi_utils::root){std::cerr<<"Error parsing arguments. Aborting...\n";}
            exit(1);
        }
    }
    //positional arguments
    if((argc-optind)<2){
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    size_t q=(size_t) atoi(argv[optind++]);
    size_t n_samples=(size_t) atoi(argv[optind++]);
    std::vector<size_t> ls;
    if(n_samples==0){ //input file mode
        std::cout<<"n_samples must be greater than 0\n";
        exit(1);
    }
    //check pos arg counts
    size_t d;
    if((argc-optind)>0){
        d=(size_t) atoi(argv[optind++]);
    }
    else{
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    if((argc-optind)==4){
        size_t l=(size_t) atoi(argv[optind++]);
        if((init_tree_type=="pbttn")&&(!(((l&(l-1))==0)&&(l!=0)))){
            std::cout<<"Dimensions must be a power of 2.\n";
            exit(1);
        }
        if((!open_bc)&&(l<3)){
            if(mpi_utils::root){
                std::cerr<<"Error: dimension length must be at least 3 if periodic bcs are used to ensure that the graph has no multiedges.\n";
                print_usage();
            }
            exit(1);
        }
        for(size_t i=0;i<d;i++){
            ls.push_back(l);
        }
    }
    else if((argc-optind)==(d+3)){
        for(size_t i=0;i<d;i++){
            size_t l=(size_t) atoi(argv[optind++]);
            if((!open_bc)&&(l<3)){
                if(mpi_utils::root){
                    std::cerr<<"Error: dimension length must be at least 3 if periodic bcs are used to ensure that the graph has no multiedges.\n";
                    print_usage();
                }
                exit(1);
            }
            ls.push_back(l);
        }
    }
    else{
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    double min_beta=(double) atof(argv[optind++]);
    double max_beta=(double) atof(argv[optind++]);
    double step_beta=(double) atof(argv[optind++]);
    //override verbosity if proc_num>1
    if(mpi_utils::proc_num>1){
        if(mpi_utils::root && verbose>0){
            std::cerr<<"Since MPI is being used with more than 1 process, verbosity forced to 0.\n";
        }
        verbose=0;
    }
    //check presence of input file/dist/tree options
    if(!((init_tree_type=="mps")||(init_tree_type=="pbttn"))){
        if(mpi_utils::root){
            std::cerr<<"Error: -T must be one of \"mps\" or \"pbttn\".\n";
            print_usage();
        }
        exit(1);
    }
    if(n_samples!=0){
        if(!dist_set){
            if(mpi_utils::root){
                std::cerr<<"Error: if <n_samples> is not 0, -d must be supplied, while -i must not be supplied.\n";
                print_usage();
            }
            exit(1);
        }
    }
    if(dist_set){
        if(!(dist_param1_set&&dist_param2_set)){
            if(mpi_utils::root){
                std::cerr<<"Error: if -d is supplied, -1 and -2 must also be supplied, even if -2 is going to be overriden.\n";
                print_usage();
            }
            exit(1);
        }
        if(!((dist=="gaussian")||(dist=="bimodal")||(dist=="uniform"))){
            if(mpi_utils::root){
                std::cerr<<"Error: -d must be one of \"gaussian\", \"bimodal\", or \"uniform\".\n";
                print_usage();
            }
            exit(1);
        }
        if(dist=="uniform"){
            dist_param2=0;
        }
    }
    bool ti_flag=0;
    if(!no_ti){
        if((dist=="gaussian")&&(dist_param2==0)){
            ti_flag=1;
        }
        else if((dist=="bimodal")&&((dist_param1==0)||(dist_param1==1))){
            ti_flag=1;
        }
    }
    if(ti_flag){
        std::cout<<"Due to settings, enforcing translational invariance in distribution function calculation.\n";
    }
    if(r_max==0){
        std::cout<<"Maximum rank is zero. Will default to r_max=q.\n";
        r_max=q;
    }
    std::vector<double> times;
    stopwatch sw,sw_total;
    sw_total.start();
    bool add_suffix=(n_samples==0)?false:true;
    size_t n_samples_counter=(n_samples==0)?1:n_samples;
    std::vector<size_t> n_sweeps_vec;
    for(size_t sweep=min_n_sweeps;sweep<=max_n_sweeps;sweep+=step_n_sweeps){
        n_sweeps_vec.push_back(sweep);
    }
    for(size_t sample=mpi_utils::proc_rank;sample<n_samples_counter;sample+=mpi_utils::proc_num){
        std::string sample_output_fn=output;
        std::string sample_mc_output_fn=output+"_mc";
        std::string sample_ar_output_fn=output+"_ar";
        std::string sample_overlaps_output_fn=output+"_overlaps";
        if(n_samples!=0){
            if(verbose>=1){std::cout<<"sample "<<sample<<":\n";}
            if(add_suffix){
                sample_output_fn+="_"+std::to_string(sample);
                sample_mc_output_fn+="_"+std::to_string(sample);
                sample_ar_output_fn+="_"+std::to_string(sample);
                sample_overlaps_output_fn+="_"+std::to_string(sample);
            }
        }
        sample_output_fn+=".txt";
        sample_mc_output_fn+=".txt";
        sample_ar_output_fn+=".dat";
        sample_overlaps_output_fn+=".dat";
        double beta=min_beta;
        std::stringstream header1_ss,header1_vals_ss,header2_ss,header2_mc_ss;
        observables::output_lines.clear(); //flush output lines
        observables::mc_output_lines.clear(); //flush output lines
        std::vector<std::pair<double,std::vector<double> > > acceptance_ratio_data;
        std::string header1_ls_str;
        if(ls.empty()){
            header1_ls_str+="fn";
        }
        else{
            for(size_t i=0;i<ls.size();i++){
                header1_ls_str+="l"+std::to_string(i)+" ";
            }
            header1_ls_str+="dist param1 param2";
        }
        header1_ss<<"idx q d r "<<header1_ls_str<<"\n";
        std::string header1_ls_vals_str;
        if(ls.empty()){
            header1_ls_vals_str+="\""+input+"\"";
        }
        else{
            for(size_t i=0;i<ls.size();i++){
                header1_ls_vals_str+=std::to_string(ls[i])+" ";
            }
            header1_ls_vals_str+=dist+" "+std::to_string(dist_param1)+" "+std::to_string(dist_param2);
        }
        header1_vals_ss<<sample<<" "<<q<<" "<<ls.size()<<" "<<r_max<<" "<<header1_ls_vals_str<<"\n";
        if(n_samples!=0){ //hypercubic lattice is used
            header2_ss<<"idx c q d r "<<header1_ls_str<<" beta m1_1_abs m1_2_abs m2_1 m2_2 m4_1 m4_2 q2 q4 q2_std sus_fm sus_sg binder_m binder_q corr_len_sg total_c\n";
        }
        else{
            header2_ss<<"idx c q d r "<<header1_ls_str<<" beta m1_1_abs m1_2_abs m2_1 m2_2 m4_1 m4_2 q2 q4 q2_std sus_fm sus_sg binder_m binder_q total_c\n";
        }
        // header2_mc_ss<<"idx c w n q d r "<<header1_ls_str<<" beta m1_abs_mean m1_abs_sd m2_mean m2_sd m4_mean m4_sd e1_mean e1_sd e2_mean e2_sd sus_fm_mean sus_fm_sd binder_m_mean binder_m_sd c_mean c_sd\n";
        // header2_mc_ss<<"idx c w n q d r "<<header1_ls_str<<" beta m1_abs_mean m1_abs_sd m2_mean m2_sd m4_mean m4_sd q1_abs_mean q1_abs_sd q2_mean q2_sd q4_mean q4_sd e1_mean e1_sd e2_mean e2_sd sus_fm_mean sus_fm_sd sus_sg_mean sus_sg_sd binder_m_mean binder_m_sd binder_q_mean binder_q_sd c_mean c_sd\n";
        header2_mc_ss<<"idx c w n q d r "<<header1_ls_str<<" beta m1_abs_mean m1_abs_sd m2_mean m2_sd m4_mean m4_sd q1_abs_mean q1_abs_sd q2_mean q2_sd q4_mean q4_sd e1_mean e1_sd e2_mean e2_sd sus_fm_mean sus_fm_sd sus_sg_mean sus_sg_sd binder_m_mean binder_m_sd binder_q_mean binder_q_sd c_mean c_sd mi_mean mi_sd\n";
        observables::output_lines.push_back(header2_ss.str());
        observables::mc_output_lines.push_back(header2_mc_ss.str());
        do{
            //flush caches for observable computation
            observables::m_known_factors.clear();
            observables::m_known_factors_complex.clear();
            observables::q_known_factors.clear();
            observables::q_known_factors_complex.clear();
            double trial_time=0; //not including init time
            if(verbose>=2){std::cout<<((use_t)?"temp=":"beta=")<<beta<<"\n";}

            graph<bmi_comparator> g=gen_lattice<bmi_comparator>(q,r_max,ls,open_bc,dist,dist_param1,dist_param2,((use_t)?1/beta:beta),init_tree_type);
            //MC observables
            std::vector<double> acceptance_ratios;
            double acceptance_ratio;
            sw.start();
            for(auto it=g.es().begin();it!=g.es().end();++it){
                bond current=*it;
                algorithm::calculate_site_probs(g,current);
        
                size_t r_k=g.vs()[current.order()].rank();
                g.vs()[current.order()].m_vec()=std::vector<std::vector<double> >();
                for(size_t idx=0;idx<r_k;idx++){
                    std::vector<double> res(r_k-1,0);
                    double prob_factor=g.vs()[current.order()].p_k()[idx];
                    for(size_t i=0;i<r_k;i++){
                        std::vector<double> contrib=observables::m_vec(g,current.order(),i,idx,0);
                        for(size_t j=0;j<contrib.size();j++){
                            res[j]+=prob_factor*contrib[j];
                        }
                    }
                    g.vs()[current.order()].m_vec().push_back(res);
                }
            }
            sampling::mh_sample(g,1000,acceptance_ratio,rand_mc);
            acceptance_ratios.push_back(acceptance_ratio);
            sw.split();
            if(verbose>=3){std::cout<<"mc sampling time: "<<(double) sw.elapsed()<<"ms\n";}
            trial_time+=sw.elapsed();
            sw.reset();
            
            sw.start();
            size_t n_samples_per_mc=1000;
            size_t n_mc_repeats=100;
            observables::calc_tree_observables(g,sample,0,q,ls.size(),r_max,((use_t)?1/beta:beta),header1_ls_vals_str,(g.dims().size()!=0));
            observables::calc_mc_observables(g,sample,0,q,ls.size(),r_max,((use_t)?1/beta:beta),header1_ls_vals_str,n_samples_per_mc,n_sweeps_vec,n_mc_repeats,rand_mc,ti_flag); //last argument is 1 to enforce translational invariance
            sw.split();
            if(verbose>=3){std::cout<<"observable computation time: "<<(double) sw.elapsed()<<"ms\n";}
            trial_time+=sw.elapsed();
            sw.reset();
            for(size_t c=0;c<n_cycles;c++){ //perform MC sampling if n_cycles>0
                std::cout<<"cycle "<<(c+1)<<"\n";
                sw.start();
                algorithm::train_nll(g,n_config_samples,100,n_nll_iter_max); //nll training
                sw.split();
                if(verbose>=3){std::cout<<"nll training time: "<<(double) sw.elapsed()<<"ms\n";}
                trial_time+=sw.elapsed();
                sw.reset();
                
                sw.start();
                sampling::mh_sample(g,1000,acceptance_ratio,rand_mc);
                acceptance_ratios.push_back(acceptance_ratio);
                sw.split();
                if(verbose>=3){std::cout<<"mc sampling time: "<<(double) sw.elapsed()<<"ms\n";}
                trial_time+=sw.elapsed();
                sw.reset();
                
                sw.start();
                observables::calc_tree_observables(g,sample,c+1,q,ls.size(),r_max,((use_t)?1/beta:beta),header1_ls_vals_str,(g.dims().size()!=0));
                observables::calc_mc_observables(g,sample,c+1,q,ls.size(),r_max,((use_t)?1/beta:beta),header1_ls_vals_str,n_samples_per_mc,n_sweeps_vec,n_mc_repeats,rand_mc,ti_flag); //last argument is 1 to enforce translational invariance
                sw.split();
                if(verbose>=3){std::cout<<"observable computation time: "<<(double) sw.elapsed()<<"ms\n";}
                trial_time+=sw.elapsed();
                sw.reset();
            }
            acceptance_ratio_data.push_back(std::make_pair(beta,acceptance_ratios));
            
            //compute output quantities
            if(verbose>=4){std::cout<<std::string(g);}
            if(verbose>=2){std::cout<<"Time elapsed for this beta/temp: "<<trial_time<<"ms\n";}
            times.push_back(trial_time);
            beta+=step_beta;
        }
        while(beta<=max_beta);
        if(output_set){
            observables::write_output(sample_output_fn,observables::output_lines);
            if(acceptance_ratio_data.size()>0){observables::write_binary_output(sample_ar_output_fn,acceptance_ratio_data);}
            observables::write_output(sample_mc_output_fn,observables::mc_output_lines);
        }
        else{
            observables::write_output(observables::output_lines);
            if(acceptance_ratio_data.size()>0){observables::write_binary_output(acceptance_ratio_data);}
            observables::write_output(observables::mc_output_lines);
        }
    }
    sw_total.split();
    //time statistics
    double timer_mean=0;
    double timer_std=0;
    for(size_t i=0;i<times.size();i++){
        timer_mean+=times[i];
    }
    timer_mean/=times.size();
    for(size_t i=0;i<times.size();i++){
        timer_std+=pow(times[i]-timer_mean,2);
    }
    timer_std/=(times.size()-1);
    timer_std=sqrt(timer_std);
    if(verbose>=1){std::cout<<"Avg. time/trial: "<<timer_mean<<"ms+/-"<<timer_std<<"ms\n";}
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpi_utils::root&&(verbose>=1)){std::cout<<"Total time: "<<(double) sw_total.elapsed()<<"ms\n";}
    MPI_Finalize();
    return 0;
}