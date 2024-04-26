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
#include "algorithm.hpp"
#include "observables.hpp"

#define PI 3.14159265358979323846

void print_usage(){
    std::cerr<<"usage: cmd/renyi_approx [--options] <q> <n_samples> <d> <{l|l0,l1,...}> <min_beta> <max_beta> <step_beta>\n";
    std::cerr<<"usage: cmd/renyi_approx [--options] <q> 0 <min_beta> <max_beta> <step_beta>\n";
    std::cerr<<"if <n_samples>!=0, required: -d, -1, -2, forbidden: -i,\n";
    std::cerr<<"if <n_samples>==0, reguired: -i, forbidden: -d, -1, -2, --open-bc\n";
    std::cerr<<"options:\n";
    std::cerr<<"\t--open-bc: generate graph connectivity with open boundary conditions.\n";
    std::cerr<<"\t--use-t: min_beta, max_beta, and step_beta refer to temperature instead.\n";
    std::cerr<<"\t-h,--help: display this message\n";
    std::cerr<<"\t-v,--verbose:\n\t\t0->nothing printed to stdout (forced for MPI)\n\t\t1->sample number and aggregate timing data\n\t\t2->per-instance timing\n\t\t3->more detailed timing breakdown\n\t\t4->graph contents, debug observable data\n";
    std::cerr<<"\t-i,--input: path to specified input file containing graph description. the graph is assumed to not contain multiedges.\n";
    std::cerr<<"\t-o,--output: prefix for output files. please omit the file extension.\n";
    std::cerr<<"\t-d,--distribution: distribution for sampling bond configurations. one of \"gaussian\",\"bimodal\" (+1/-1),\"uniform\".\n";
    std::cerr<<"\t-1,--dist-param1: distribution hyperparameter.\n\t\tgaussian-> mean\n\t\tbimodal -> probability of ferromagnetic bond\n\t\tuniform -> minimum bond strength\n";
    std::cerr<<"\t-2,--dist-param2: distribution hyperparameter.\n\t\tgaussian-> standard deviation\n\t\tbimodal -> ignored, overriden to 0\n\t\tuniform -> maximum bond strength\n";
    std::cerr<<"\t-r,--r-max: maximum rank of spins in the approximation\n";
    std::cerr<<"\t-n,--iter-max: maximum number of optimization iterations\n";
    std::cerr<<"\t-l,--learning-rate: learning rate. if nonzero, the optimization method will be gradient descent instead of iterative optimization.\n";
    std::cerr<<"\t-R,--restarts: maximum number of restarts\n";
    std::cerr<<"\t-c,--configs: number of configs to sample per temperature\n";
}

template<typename cmp>
graph<cmp> gen_lattice(size_t q,std::vector<size_t> ls,bool open_bc,std::string dist_type,double dist_param1,double dist_param2,double beta){ //transformations are done to counteract the transformations in gen_hypercubic
    graph<cmp> g;
    if(dist_type=="gaussian"){
        //dist_param1=mean, dist_param2=std
        std::normal_distribution<double> dist((dist_param1+1)/2,dist_param2/2);
        g=graph_utils::gen_hypercubic<std::normal_distribution<double>,cmp>(q,ls,!open_bc,dist,beta);
    }
    else if(dist_type=="bimodal"){
        //dist_param1=p, dist_param2=N/A
        std::discrete_distribution<int> dist{1-dist_param1,dist_param1};
        g=graph_utils::gen_hypercubic<std::discrete_distribution<int>,cmp>(q,ls,!open_bc,dist,beta);
    }
    else if(dist_type=="uniform"){
        //dist_param1=min, dist_param2=max
        std::uniform_real_distribution<double> dist{(dist_param1+1)/2,(dist_param2+1)/2};
        g=graph_utils::gen_hypercubic<std::uniform_real_distribution<double>,cmp>(q,ls,!open_bc,dist,beta);
    }
    return g;
}

template<typename cmp>
void calc_observables(graph<cmp>& g,size_t q_orig,double& m1_1_abs,double& m1_2_abs,double& m2_1,double& m2_2,double& m4_1,double& m4_2,double& q2,double& q4){
    //use bottom-up approach to compute observables, avoiding stack overflow
    q2=observables::q(g,q_orig,g.vs().size()-1,2,2,0)/pow(g.n_phys_sites(),2);
    q4=observables::q(g,q_orig,g.vs().size()-1,4,2,0)/pow(g.n_phys_sites(),4);
    m1_1_abs=observables::m(g,q_orig,g.vs().size()-1,1,1,1)/g.n_phys_sites();
    m1_2_abs=observables::m(g,q_orig,g.vs().size()-1,1,2,1)/g.n_phys_sites();
    m2_1=observables::m(g,q_orig,g.vs().size()-1,2,1,0)/pow(g.n_phys_sites(),2);
    m2_2=observables::m(g,q_orig,g.vs().size()-1,2,2,0)/pow(g.n_phys_sites(),2);
    m4_1=observables::m(g,q_orig,g.vs().size()-1,4,1,0)/pow(g.n_phys_sites(),4);
    m4_2=observables::m(g,q_orig,g.vs().size()-1,4,2,0)/pow(g.n_phys_sites(),4);
}

template<typename cmp>
void calc_observables(graph<cmp>& g,size_t q_orig,double& m1_1_abs,double& m1_2_abs,double& m2_1,double& m2_2,double& m4_1,double& m4_2,double& q2,double& q4,double& k_min,std::complex<double>& q2_k){
    //use bottom-up approach to compute observables, avoiding stack overflow
    q2=observables::q(g,q_orig,g.vs().size()-1,2,2,0)/pow(g.n_phys_sites(),2);
    q4=observables::q(g,q_orig,g.vs().size()-1,4,2,0)/pow(g.n_phys_sites(),4);
    m1_1_abs=observables::m(g,q_orig,g.vs().size()-1,1,1,1)/g.n_phys_sites();
    m1_2_abs=observables::m(g,q_orig,g.vs().size()-1,1,2,1)/g.n_phys_sites();
    m2_1=observables::m(g,q_orig,g.vs().size()-1,2,1,0)/pow(g.n_phys_sites(),2);
    m2_2=observables::m(g,q_orig,g.vs().size()-1,2,2,0)/pow(g.n_phys_sites(),2);
    m4_1=observables::m(g,q_orig,g.vs().size()-1,4,1,0)/pow(g.n_phys_sites(),4);
    m4_2=observables::m(g,q_orig,g.vs().size()-1,4,2,0)/pow(g.n_phys_sites(),4);
    if(g.dims().size()!=0){
        std::vector<double> k(g.dims().size(),0);
        auto max_it=std::max_element(g.dims().begin(),g.dims().end());
        size_t max_idx=max_it-g.dims().begin();
        k_min=2*PI/(*max_it); //max dim yields min k
        k[max_idx]=k_min;
        q2_k=observables::m(g,q_orig,g.vs().size()-1,2,2,k,0)/pow(g.n_phys_sites(),2);
    }
}

int main(int argc,char **argv){
    //mpi init
    mpi_utils::init();
    //argument handling
    int open_bc=0;
    int use_t=0;
    size_t verbose=0;
    bool input_set=false;
    bool output_set=false;
    bool dist_set=false;
    bool dist_param1_set=false;
    bool dist_param2_set=false;
    std::string input,output,dist;
    double dist_param1,dist_param2;
    size_t r_max=0;
#ifdef MODEL_CMD
    size_t iter_max=100; //default is 100 iterations max
    double lr=0;
    size_t restarts=1;
#endif
#ifdef MODEL_RENYI
    size_t iter_max=100000; //default is 100000 iterations max
    double lr=0.0001;
    size_t restarts=10;
#endif
#ifdef MODEL_CPD
    size_t iter_max=100; //default is 100 iterations max
    double lr=0;
    size_t restarts=1;
#endif
    size_t n_configs=0;
    //option arguments
    while(1){
        static struct option long_opts[]={
            {"open-bc",no_argument,&open_bc,1},
            {"use-t",no_argument,&use_t,1},
            {"help",no_argument,0,'h'},
            {"verbose",required_argument,0,'v'},
            {"input",required_argument,0,'i'},
            {"output",required_argument,0,'o'},
            {"distribution",required_argument,0,'d'},
            {"dist-param1",required_argument,0,'1'},
            {"dist-param2",required_argument,0,'2'},
            {"r-max",required_argument,0,'r'},
            {"iter-max",required_argument,0,'n'},
            {"learning-rate",required_argument,0,'l'},
            {"restarts",required_argument,0,'R'},
            {"configs",required_argument,0,'c'},
            {0, 0, 0, 0}
        };
        int opt_idx=0;
        int c=getopt_long(argc,argv,"hv:i:o:d:1:2:r:n:l:R:c:",long_opts,&opt_idx);
        if(c==-1){break;} //end of options
        switch(c){
            //handle long option flags
            case 0:
            break;
            //handle standard options
            case 'h': print_usage(); exit(1);
            case 'v': verbose=(size_t) atoi(optarg); break;
            case 'i': input=std::string(optarg); input_set=true; break;
            case 'o': output=std::string(optarg); output_set=true; break;
            case 'd': dist=std::string(optarg); dist_set=true; break;
            case '1': dist_param1=(double) atof(optarg); dist_param1_set=true; break;
            case '2': dist_param2=(double) atof(optarg); dist_param2_set=true; break;
            case 'r': r_max=(size_t) atoi(optarg); break;
            case 'n': iter_max=(size_t) atoi(optarg); break;
            case 'l': lr=(double) atof(optarg); break;
            case 'R': restarts=(size_t) atoi(optarg); break;
            case 'c': n_configs=(size_t) atoi(optarg); break;
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
        if((argc-optind)!=3){
            if(mpi_utils::root){print_usage();}
            exit(1);
        }
    }
    else{
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
    //check presence of input file/dist options
    if(n_samples==0){
        if((!input_set)||dist_set||dist_param1_set||dist_param2_set||open_bc){
            if(mpi_utils::root){
                std::cerr<<"Error: if <n_samples> is 0, -i must be supplied, while -d, -1, -2, --open-bc must not be supplied.\n";
                print_usage();
            }
            exit(1);
        }
    }
    if(n_samples!=0){
        if(input_set||(!dist_set)){
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
    if(r_max==0){
        std::cout<<"Maximum rank is zero. Will default to r_max=q.\n";
    }
    if(lr!=0){
        std::cout<<"Learning rate is nonzero. Using gradient descent method.\n";
    }
    std::vector<double> times;
    stopwatch sw,sw_total;
    sw_total.start();
    bool add_suffix=(n_samples==0)?false:true;
    size_t n_samples_counter=(n_samples==0)?1:n_samples;
    for(size_t sample=mpi_utils::proc_rank;sample<n_samples_counter;sample+=mpi_utils::proc_num){
        std::string sample_output_fn=output;
        std::string sample_overlaps_output_fn=output;
        if(n_samples!=0){
            if(verbose>=1){std::cout<<"sample "<<sample<<":\n";}
            if(add_suffix){sample_output_fn+="_"+std::to_string(sample);}
            if(add_suffix){sample_overlaps_output_fn+="_"+std::to_string(sample);}
        }
        sample_output_fn+=".txt";
        sample_overlaps_output_fn+=".dat";
        if(n_configs>0){
            std::ofstream ofs(sample_overlaps_output_fn,std::ios::binary);
            size_t n_betas=((max_beta-min_beta)/step_beta)+1;
            ofs.write((char*) &n_betas,sizeof(n_betas));
            ofs.close();
        }
        double beta=min_beta;
        double m1_1_abs,m1_2_abs,m2_1,m2_2,m4_1,m4_2,q2,q4,k_min;
        std::complex<double> q2_k;
        double q2_var,q2_std,sus_fm,sus_sg,binder_m,binder_q,sus_sg_k,corr_len_sg;
        std::stringstream header1_ss,header1_vals_ss,header2_ss;
        observables::output_lines.clear(); //flush output lines
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
        // observables::output_lines.push_back(header1_vals_ss.str());
        if(n_samples!=0){ //hypercubic lattice is used
            header2_ss<<"idx q d r "<<header1_ls_str<<" beta m1_1_abs m1_2_abs m2_1 m2_2 m4_1 m4_2 q2 q4 q2_std sus_fm sus_sg binder_m binder_q corr_len_sg total_c\n";
        }
        else{
            header2_ss<<"idx q d r "<<header1_ls_str<<" beta m1_1_abs m1_2_abs m2_1 m2_2 m4_1 m4_2 q2 q4 q2_std sus_fm sus_sg binder_m binder_q total_c\n";
        }
        observables::output_lines.push_back(header2_ss.str());
        while(beta<=max_beta){
            //flush caches for observable computation
            observables::m_known_factors.clear();
            observables::m_known_factors_complex.clear();
            observables::q_known_factors.clear();
            observables::q_known_factors_complex.clear();
            double trial_time=0; //not including init time
            if(verbose>=2){std::cout<<((use_t)?"temp=":"beta=")<<beta<<"\n";}
            size_t n_phys_sites;

            graph<bmi_comparator> g=input_set?graph_utils::load_graph<bmi_comparator>(input,q,((use_t)?1/beta:beta)):gen_lattice<bmi_comparator>(q,ls,open_bc,dist,dist_param1,dist_param2,((use_t)?1/beta:beta));
            sw.start();
            algorithm::approx(q,g,r_max,iter_max,lr,restarts);
            sw.split();
            if(verbose>=3){std::cout<<"approx time: "<<(double) sw.elapsed()<<"ms\n";}
            trial_time+=sw.elapsed();
            sw.reset();
            if(n_configs>0){
                sw.start();
                std::vector<std::vector<size_t> > samples=sampling::sample(g,n_configs);
                std::vector<double> overlaps=sampling::pair_overlaps(samples,q);
                sw.split();
                if(verbose>=3){std::cout<<"config sampling time: "<<(double) sw.elapsed()<<"ms\n";}
                trial_time+=sw.elapsed();
                sw.reset();
                if(output_set){
                    sampling::write_output(sample_overlaps_output_fn,overlaps,beta);
                }
                else{
                    sampling::write_output(overlaps);
                }
            }
            sw.start();
            if(g.dims().size()!=0){
                calc_observables(g,q,m1_1_abs,m1_2_abs,m2_1,m2_2,m4_1,m4_2,q2,q4,k_min,q2_k);
            }
            else{
                calc_observables(g,q,m1_1_abs,m1_2_abs,m2_1,m2_2,m4_1,m4_2,q2,q4);
            }
            sw.split();
            if(verbose>=3){std::cout<<"observable computation time: "<<(double) sw.elapsed()<<"ms\n";}
            trial_time+=sw.elapsed();
            sw.reset();
            if(verbose>=4){observables::print_moments(g,q);}
            if(verbose>=4){std::cout<<std::string(g);}
            n_phys_sites=g.n_phys_sites();
            
            //compute cumulative cost
            double total_cost=0;
            for (auto it=g.es().begin();it!=g.es().end();++it){
                total_cost+=(*it).cost();
            }
            //compute output quantities
            q2_var=q4-pow(q2,2);
            q2_std=sqrt(q2_var);
            sus_fm=n_phys_sites*m1_2_abs;
            sus_sg=n_phys_sites*q2;
            binder_m=0.5*(3-(m4_1/pow(m2_1,2)));
            binder_q=0.5*(3-(q4/pow(q2,2)));
            if(add_suffix){ //hypercubic lattice is used
                sus_sg_k=n_phys_sites*sqrt(std::norm(q2_k));
                corr_len_sg=sqrt((sus_sg/sus_sg_k)-1)/(2*sin(k_min/2));
            }
            //prepare output lines
            std::stringstream output_line_ss;
            if(add_suffix){ //hypercubic lattice is used
                output_line_ss<<std::scientific<<sample<<" "<<q<<" "<<ls.size()<<" "<<r_max<<" "<<header1_ls_vals_str<<" "<<((use_t)?1/beta:beta)<<" "<<m1_1_abs<<" "<<m1_2_abs<<" "<<m2_1<<" "<<m2_2<<" "<<m4_1<<" "<<m4_2<<" "<<q2<<" "<<q4<<" "<<q2_std<<" "<<sus_fm<<" "<<sus_sg<<" "<<binder_m<<" "<<binder_q<<" "<<corr_len_sg<<" "<<total_cost<<"\n";
            }
            else{
                output_line_ss<<std::scientific<<sample<<" "<<q<<" "<<ls.size()<<" "<<r_max<<" "<<header1_ls_vals_str<<" "<<((use_t)?1/beta:beta)<<" "<<m1_1_abs<<" "<<m1_2_abs<<" "<<m2_1<<" "<<m2_2<<" "<<m4_1<<" "<<m4_2<<" "<<q2<<" "<<q4<<" "<<q2_std<<" "<<sus_fm<<" "<<sus_sg<<" "<<binder_m<<" "<<binder_q<<" "<<total_cost<<"\n";
            }
            observables::output_lines.push_back(output_line_ss.str());
            if(verbose>=2){std::cout<<"Time elapsed for this beta/temp: "<<trial_time<<"ms\n";}
            times.push_back(trial_time);
            beta+=step_beta;
        }
        if(output_set){
            observables::write_output(sample_output_fn,observables::output_lines);
        }
        else{
            observables::write_output(observables::output_lines);
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