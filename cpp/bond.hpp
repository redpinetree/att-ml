#ifndef BOND
#define BOND

#include <iostream>
#include <string>
#include <utility>

#include "ndarray.hpp"

class bond{
public:
    bond();
    bond(int,int,int,array3d<double>);
    bond(std::pair<int,int>,int,array3d<double>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const bond&);
    std::pair<int,int> v() const;
    int v1() const;
    int v2() const;
    int depth() const;
    const array3d<double>& w() const;
    double bmi() const;
    double ee() const;
    int order() const;
    bool todo() const;
    std::pair<int,int>& v();
    int& v1();
    int& v2();
    int& depth();
    array3d<double>& w();
    double& bmi();
    double& ee();
    int& order();
    bool& todo();
    void bmi(array3d<double>&);
    void save(std::ostream& os);
    static bond load(std::istream& is);
private:
    bool todo_; //if not yet processed, todo=1
    std::pair<int,int> v_;
    int depth_;
    int order_; //in observable computation, this is the upstream site
    double bmi_;
    double ee_; //for born machine, should be the same as bmi in single-layer scheme
    array3d<double> w_;
};

#endif