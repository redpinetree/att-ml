#include "mpi.h"
#include <random>
#include <chrono>

#include "mpi_utils.hpp"

int mpi_utils::proc_num;
int mpi_utils::proc_rank;
bool mpi_utils::root;
std::mt19937_64 mpi_utils::prng;

void mpi_utils::init(){
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_utils::proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_utils::proc_rank);
    std::random_device device;
    std::seed_seq seq{device(),device(),device(),device()};
    // mpi_utils::prng.seed(seq);
    // mpi_utils::prng.seed(0);
    prng.seed((size_t) std::chrono::system_clock::now().time_since_epoch().count()+proc_rank);
    mpi_utils::root=(proc_rank==0);
}
