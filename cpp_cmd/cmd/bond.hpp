#ifndef BOND
#define BOND

#include <iostream>
#include <string>
#include <utility>

#include "../ndarray.hpp"

class bond{
public:
    bond();
    bond(size_t,size_t,array2d<double>);
    bond(std::pair<size_t,size_t>,array2d<double>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const bond&);
    std::pair<size_t,size_t> v() const;
    std::pair<size_t,size_t> v_orig() const;
    size_t v1() const;
    size_t v2() const;
    size_t v1_orig() const;
    size_t v2_orig() const;
    size_t virt_count() const;
    array2d<double> w() const;
    array2d<size_t> f() const;
    double bmi() const;
    size_t order() const;
    bool todo() const;
    std::pair<size_t,size_t>& v();
    std::pair<size_t,size_t>& v_orig();
    size_t& v1();
    size_t& v2();
    size_t& v1_orig();
    size_t& v2_orig();
    size_t& virt_count();
    array2d<double>& w();
    array2d<size_t>& f();
    double& bmi();
    size_t& order();
    bool& todo();
    void bmi(array2d<double>&);
private:
    std::pair<size_t,size_t> v_;
    std::pair<size_t,size_t> v_orig_;
    size_t virt_count_;
    array2d<double> w_; //if not yet processed, todo=1
    array2d<size_t> f_;
    double bmi_;
    size_t order_; //in observable computation, this is the upstream site
    bool todo_;
};

#endif