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

#ifndef STOPWATCH
#define STOPWATCH

#include <chrono>

class stopwatch{
public:
  stopwatch():start_(std::chrono::high_resolution_clock::now()),total_(0){};
  void start(){
    this->start_=std::chrono::high_resolution_clock::now();
  };
  void split(){
    this->total_+=std::chrono::high_resolution_clock::now()-this->start_;
  };
  void reset(){
    this->total_=std::chrono::high_resolution_clock::duration(0);
  };
  std::chrono::high_resolution_clock::duration::rep elapsed() const{
    return std::chrono::duration_cast<std::chrono::milliseconds>(this->total_).count();
  };
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::high_resolution_clock::duration total_;
};

#endif
