#include <cmath>
#include <tuple>

#include "bond.hpp"

struct vertices_comparator{
    explicit vertices_comparator(){}
    
    bool operator()(const bond& e1,const bond& e2) const{
        return e1.v()<e2.v();
    }
};

struct coupling_comparator{
    explicit coupling_comparator(){}
    explicit coupling_comparator(size_t q_){}
    
    bool operator()(const bond& e1,const bond& e2) const{
        return std::make_tuple(e1.todo(),fabs(e1.j()),e1.v())<std::make_tuple(e2.todo(),fabs(e2.j()),e2.v());
    }
    
    size_t q;
};

struct bmi_comparator{
    explicit bmi_comparator(){}
    explicit bmi_comparator(size_t q_): q(q_){}
    
    bool operator()(const bond& e1,const bond& e2) const{
        return std::make_tuple(e1.todo(),e1.bmi(),e1.v())<std::make_tuple(e2.todo(),e2.bmi(),e2.v());
    }
    
    size_t q;
};