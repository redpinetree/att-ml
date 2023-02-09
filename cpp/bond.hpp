#ifndef BOND
#define BOND

#include <iostream>
#include <string>
#include <utility>

class bond{
public:
    bond(size_t,size_t,size_t,double);
    bond(size_t,std::pair<size_t,size_t>,double);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const bond&);
    size_t q() const;
    std::pair<size_t,size_t> v() const;
    size_t v1() const;
    size_t v2() const;
    double j() const;
    double bmi() const;
    size_t order() const;
    bool todo() const;
    size_t& q();
    std::pair<size_t,size_t>& v();
    size_t& v1();
    size_t& v2();
    double& j();
    double& bmi();
    size_t& order();
    bool& todo();
    void bmi(size_t,double);
private:
    size_t q_;
    std::pair<size_t,size_t> v_;
    double j_;
    double bmi_;
    size_t order_;
    bool todo_;
};

#endif