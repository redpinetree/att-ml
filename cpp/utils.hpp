/*
Copyright 2025 Katsuya O. Akamatsu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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