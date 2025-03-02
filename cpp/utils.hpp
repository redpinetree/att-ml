#ifndef UTILS_
#define UTILS_

#include <random>
#include <tuple>

#include "bond.hpp"

extern std::mt19937_64 prng;

struct bmi_comparator{
    explicit bmi_comparator(){}
    explicit bmi_comparator(int q_): q(q_){}
    
    inline bool operator()(const bond& e1,const bond& e2) const{
        // return std::make_tuple(e1.todo(),e1.bmi(),e1.v())<std::make_tuple(e2.todo(),e2.bmi(),e2.v());
        return std::make_tuple(e1.todo(),e1.depth(),e1.order(),e1.bmi(),e1.v())<std::make_tuple(e2.todo(),e2.depth(),e2.order(),e2.bmi(),e2.v());
    }
    
    int q;
};

#endif