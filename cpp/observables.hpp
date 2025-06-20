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

#ifndef OBSERVABLES
#define OBSERVABLES

#include <string>
#include <utility>
#include <vector>

namespace observables{
    extern std::vector<std::string> output_lines;
    void write_output(std::string,std::vector<std::string>&);
    void write_output(std::vector<std::string>&);
    void write_binary_output(std::string,std::vector<std::pair<double,std::vector<double> > >&);
    void write_binary_output(std::vector<std::pair<double,std::vector<double> > >&);
}

#endif
