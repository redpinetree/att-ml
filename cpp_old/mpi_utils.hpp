#ifndef MPI_UTILS
#define MPI_UTILS

#include <random>
#include <chrono>

namespace mpi_utils{
    extern int proc_num;
    extern int proc_rank;
    extern bool root;
    extern std::mt19937_64 prng;
    void init();
}

#endif
