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
    std::cerr<<"usage: tree_ml [--options] <input_dim> <d> <{l|l0,l1,...}>\n";
    std::cerr<<"options:\n";
    std::cerr<<"\t-h,--help: display this message\n";
    std::cerr<<"\t-v,--verbose:\n\t\t0->nothing printed to stdout (forced for MPI)\n\t\t1->sample number and aggregate timing data\n\t\t2->per-instance timing\n\t\t3->more detailed timing breakdown\n\t\t4->graph contents, debug observable data\n";
    std::cerr<<"\t-i,--input: input file containing training data.\n";
    std::cerr<<"\t-o,--output: prefix for output files. please omit the file extension.\n";
    std::cerr<<"\t-t,--top-dim: top dimension\n";
    std::cerr<<"\t-L,--label-file: input file containing training labels\n";
    std::cerr<<"\t-T,--init-tree-type: initial tree type: mps or pbttn or rand\n";
    std::cerr<<"\t-r,--r-max: maximum rank of spins in the approximation\n";
    std::cerr<<"\t-N,--nll-iter-max: maximum number of NLL optimization iterations\n";
}

template<typename cmp>
graph<cmp> gen_graph(size_t idim,size_t tdim,size_t r_max,std::vector<size_t> ls,bool open_bc,std::string init_tree_type){ //transformations are done to counteract the transformations in gen_hypercubic
    graph<cmp> g;
    if(init_tree_type=="mps"){
        std::normal_distribution<double> dist(0,0);
        g=graph_utils::init_mps<std::normal_distribution<double>,cmp>(idim,tdim,r_max,ls,!open_bc,dist,1);
    }
    else if(init_tree_type=="pbttn"){
        std::normal_distribution<double> dist{0,0};
        g=graph_utils::init_pbttn<std::normal_distribution<double>,cmp>(idim,tdim,r_max,ls,!open_bc,dist,1);
    }
    else if(init_tree_type=="rand"){
        std::normal_distribution<double> dist{0,0};
        g=graph_utils::init_rand<std::normal_distribution<double>,cmp>(idim,tdim,r_max,ls,!open_bc,dist,1);
    }
    return g;
}

int main(int argc,char **argv){
    //mpi init
    mpi_utils::init();
    //argument handling
    int open_bc=0;
    size_t verbose=0;
    bool input_set=false;
    bool output_set=false;
    bool label_set=false;
    bool tdim_set=false;
    std::string input,output,label_file;
    std::string init_tree_type="mps";
    size_t r_max=0;
    size_t tdim=0;
    size_t n_config_samples=1000;
    size_t n_nll_iter_max=10000;
    double lr=0.001;
    //option arguments
    while(1){
        static struct option long_opts[]={
            {"open-bc",no_argument,&open_bc,1},
            {"help",no_argument,0,'h'},
            {"verbose",required_argument,0,'v'},
            {"input",required_argument,0,'i'},
            {"output",required_argument,0,'o'},
            {"top-dim",required_argument,0,'t'},
            {"label-file",required_argument,0,'L'},
            {"init-tree-type",required_argument,0,'T'},
            {"r-max",required_argument,0,'r'},
            {"nll-iter-max",required_argument,0,'N'},
            {"learning-rate",required_argument,0,'l'},
            {0, 0, 0, 0}
        };
        int opt_idx=0;
        int c=getopt_long(argc,argv,"hv:i:o:t:L:T:r:N:l:",long_opts,&opt_idx);
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
            case 't': tdim=(size_t) atoi(optarg); tdim_set=true; break;
            case 'L': label_file=std::string(optarg); label_set=true; break;
            case 'T': init_tree_type=std::string(optarg); break;
            case 'r': r_max=(size_t) atoi(optarg); break;
            case 'N': n_nll_iter_max=(size_t) atoi(optarg); break;
            case 'l': lr=atof(optarg); break;
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
    size_t idim=(size_t) atoi(argv[optind++]);
    std::vector<size_t> ls;
    //check pos arg counts
    size_t d;
    if((argc-optind)>0){
        d=(size_t) atoi(argv[optind++]);
    }
    else{
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    if((argc-optind)==1){
        size_t l=(size_t) atoi(argv[optind++]);
        if((init_tree_type=="pbttn")&&(!(((l&(l-1))==0)&&(l!=0)))){
            std::cout<<"Dimensions must be a power of 2.\n";
            exit(1);
        }
        for(size_t i=0;i<d;i++){
            ls.push_back(l);
        }
    }
    else if((argc-optind)==d){
        for(size_t i=0;i<d;i++){
            size_t l=(size_t) atoi(argv[optind++]);
            if((init_tree_type=="pbttn")&&(!(((l&(l-1))==0)&&(l!=0)))){
                std::cout<<"Dimensions must be a power of 2.\n";
                exit(1);
            }
            ls.push_back(l);
        }
    }
    else{
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    //check if input specified
    if(!input_set){
        if(mpi_utils::root){
            std::cerr<<"Error: Must specify input file with training data.\n";
            print_usage();
        }
        exit(1);
    }
    //override verbosity if proc_num>1
    if(mpi_utils::proc_num>1){
        if(mpi_utils::root && verbose>0){
            std::cerr<<"Since MPI is being used with more than 1 process, verbosity forced to 0.\n";
        }
        verbose=0;
    }
    //check initial tree options
    if(!((init_tree_type=="mps")||(init_tree_type=="pbttn")||(init_tree_type=="rand"))){
        if(mpi_utils::root){
            std::cerr<<"Error: -T must be one of \"mps\" or \"pbttn\" or \"rand\".\n";
            print_usage();
        }
        exit(1);
    }
    //check if exactly one of top dim and label file is specified
    if(tdim_set&&label_set){
        if(mpi_utils::root){
            std::cerr<<"Error: -t cannot be set together with -L.\n";
            print_usage();
        }
        exit(1);
    }
    if(r_max==0){
        std::cout<<"Maximum rank not specified/is zero. Will default to r_max=input_dim.\n";
        r_max=idim;
    }
    std::vector<double> times;
    stopwatch sw,sw_total;
    sw_total.start();
    
    std::string output_fn=output;
    std::string samples_fn=output+"_samples";
    output_fn+=".txt";
    samples_fn+=".txt";
    std::stringstream header1_ss,header1_vals_ss,header2_ss,header2_mc_ss;
    std::string header1_ls_str;
    observables::output_lines.clear(); //flush output lines
    std::string header1_ls_vals_str;
    for(size_t i=0;i<ls.size();i++){
        header1_ls_str+="l"+std::to_string(i)+" ";
        header1_ls_vals_str+=std::to_string(ls[i])+" ";
    }
    header1_ss<<"idim r fn d \n";
    header1_vals_ss<<idim<<" "<<r_max<<" "<<("\""+input)<<" "<<ls.size()<<" "<<header1_ls_vals_str<<"\n";
    observables::output_lines.push_back(header1_ss.str());
    
    double trial_time=0; //not including init time
    
    size_t train_data_idim,n_samples,train_data_total_length;
    std::vector<sample_data> train_data=algorithm::load_training_data_from_file(input,n_samples,train_data_total_length,train_data_idim);
    std::vector<size_t> train_data_labels;
    if(label_set){
        train_data_labels=algorithm::load_training_data_labels_from_file(label_file,n_samples,tdim);
    }
    graph<bmi_comparator> g=gen_graph<bmi_comparator>(idim,tdim,r_max,ls,open_bc,init_tree_type);
    for(auto it=g.es().begin();it!=g.es().end();++it){
        bond current=*it;
        algorithm::calculate_site_probs(g,current);
    }
    
    if(g.n_phys_sites()!=train_data_total_length){
        std::cout<<"Mismatch in input site count between training data ("<<train_data_total_length<<") and model ("<<g.n_phys_sites()<<").\n";
        exit(1);
    }
    if(idim!=train_data_idim){
        std::cout<<"Mismatch in input dimension between training data ("<<train_data_idim<<") and model ("<<idim<<").\n";
        exit(1);
    }
    
    sw.start();
    if(label_set){
        algorithm::train_nll(g,train_data,train_data_labels,n_nll_iter_max,r_max,lr); //nll training with labels
    }
    else{
        algorithm::train_nll(g,train_data,n_nll_iter_max,r_max,lr); //nll training
    }
    sw.split();
    if(verbose>=3){std::cout<<"nll training time: "<<(double) sw.elapsed()<<"ms\n";}
    trial_time+=sw.elapsed();
    sw.reset();
    
    std::vector<sample_data> generated_samples=sampling::tree_sample(g,10);
    for(size_t i=0;i<generated_samples.size();i++){
        for(size_t j=0;j<generated_samples[i].n_phys_sites();j++){
            std::cout<<generated_samples[i].s()[j]<<" ";
        }
        std::cout<<"\n";
    }
    //compute output quantities
    if(verbose>=4){std::cout<<std::string(g);}
    if(verbose>=2){std::cout<<"Time elapsed: "<<trial_time<<"ms\n";}
    times.push_back(trial_time);
    
    if(output_set){
        observables::write_output(output_fn,observables::output_lines);
        // observables::write_output(mc_output_fn,observables::mc_output_lines);
    }
    else{
        observables::write_output(observables::output_lines);
        // observables::write_output(observables::mc_output_lines);
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