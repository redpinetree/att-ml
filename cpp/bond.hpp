#ifndef BOND
#define BOND

#include <iostream>
#include <string>
#include <utility>

#include "ndarray.hpp"

class bond{
public:
    bond();
    bond(size_t,size_t,size_t,array3d<double>);
    bond(std::pair<size_t,size_t>,size_t,array3d<double>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const bond&);
    std::pair<size_t,size_t> v() const;
    std::pair<size_t,size_t> v_orig() const;
    size_t v1() const;
    size_t v2() const;
    size_t v1_orig() const;
    size_t v2_orig() const;
    size_t virt_count() const;
    size_t depth() const;
    array3d<double> w() const;
    double bmi() const;
    double ee() const;
    size_t order() const;
    bool todo() const;
    std::pair<size_t,size_t>& v();
    std::pair<size_t,size_t>& v_orig();
    size_t& v1();
    size_t& v2();
    size_t& v1_orig();
    size_t& v2_orig();
    size_t& virt_count();
    size_t& depth();
    array3d<double>& w();
    double& bmi();
    double& ee();
    size_t& order();
    bool& todo();
    void bmi(array3d<double>&);
private:
    std::pair<size_t,size_t> v_;
    std::pair<size_t,size_t> v_orig_;
    size_t virt_count_;
    size_t depth_;
    array3d<double> w_;
    double bmi_;
    double ee_; //for born machine, should be the same as bmi in single-layer scheme
    size_t order_; //in observable computation, this is the upstream site
    bool todo_; //if not yet processed, todo=1
};

#endif