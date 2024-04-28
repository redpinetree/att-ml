MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -ffloat-store -ffp-contract=off -march=native -pipe -std=c++17
# CXXFLAGS = -O0

TARGET_OLD_OLD = ./bin/tree_approx_potts_old_old
TARGET_OLD = ./bin/tree_approx_potts_old
TARGET_CMD = ./bin/cmd_approx
TARGET_RENYI = ./bin/renyi_approx
TARGET_CPD = ./bin/cpd_approx

SRC_OLD_OLD = ./cpp_old_old/tree_approx_potts.cpp ./cpp_old_old/algorithm.cpp ./cpp_old_old/graph.cpp ./cpp_old_old/graph_utils.cpp ./cpp_old_old/site.cpp ./cpp_old_old/bond.cpp ./cpp_old_old/bond_utils.cpp
SRC_OLD = ./cpp_old/tree_approx_potts.cpp ./cpp_old/observables.cpp ./cpp_old/algorithm.cpp ./cpp_old/graph_utils.cpp ./cpp_old/site.cpp ./cpp_old/bond.cpp ./cpp_old/bond_utils.cpp ./cpp_old/mpi_utils.cpp
SRC_COMMON = ./cpp/main.cpp ./cpp/observables.cpp ./cpp/sampling.cpp ./cpp/site.cpp ./cpp/mpi_utils.cpp
SRC_CMD = ./cpp/cmd/graph_utils.cpp ./cpp/cmd/algorithm.cpp ./cpp/cmd/optimize.cpp ./cpp/cmd/bond.cpp
SRC_RENYI = ./cpp/renyi/graph_utils.cpp ./cpp/renyi/algorithm.cpp ./cpp/renyi/optimize.cpp ./cpp/renyi/bond.cpp
SRC_CPD = ./cpp/cpd/graph_utils.cpp ./cpp/cpd/algorithm.cpp ./cpp/cpd/optimize.cpp ./cpp/cpd/bond.cpp ./cpp/cpd/mat_ops.cpp

OBJ_OLD_OLD = $(SRC_OLD_OLD:%.cpp=%.o)
OBJ_OLD = $(SRC_OLD:%.cpp=%.o)
OBJ_COMMON_CMD = $(SRC_COMMON:%.cpp=%_cmd.o)
OBJ_COMMON_RENYI = $(SRC_COMMON:%.cpp=%_renyi.o)
OBJ_COMMON_CPD = $(SRC_COMMON:%.cpp=%_cpd.o)
OBJ_CMD = $(SRC_CMD:%.cpp=%.o)
OBJ_RENYI = $(SRC_RENYI:%.cpp=%.o)
OBJ_CPD = $(SRC_CPD:%.cpp=%.o)

.PHONY: clean graph

all: $(TARGET_CMD) $(TARGET_RENYI) $(TARGET_CPD)
cmd: $(TARGET_CMD)
renyi: $(TARGET_RENYI)
cpd: $(TARGET_CPD)
# all: $(TARGET_OLD) $(TARGET_CMD) $(TARGET_RENYI) $(TARGET_CPD)
# all: $(TARGET_OLD_OLD) $(TARGET_OLD) $(TARGET_CMD) $(TARGET_RENYI) $(TARGET_CPD)

$(TARGET_OLD_OLD): $(OBJ_OLD_OLD)
	$(CXX) $(OBJ_OLD_OLD) -o $(TARGET_OLD_OLD) $(CXXFLAGS)
	
$(TARGET_OLD): $(OBJ_OLD)
	$(MPICXX) $(OBJ_OLD) -o $(TARGET_OLD) $(CXXFLAGS)

$(TARGET_CMD): $(OBJ_COMMON_CMD) $(OBJ_CMD)
	$(MPICXX) $(OBJ_COMMON_CMD) $(OBJ_CMD) -o $(TARGET_CMD) $(CXXFLAGS)

$(TARGET_RENYI): $(OBJ_COMMON_RENYI) $(OBJ_RENYI)
	$(MPICXX) $(OBJ_COMMON_RENYI) $(OBJ_RENYI) -o $(TARGET_RENYI) $(CXXFLAGS)

$(TARGET_CPD): $(OBJ_COMMON_CPD) $(OBJ_CPD)
	$(MPICXX) $(OBJ_COMMON_CPD) $(OBJ_CPD) -llapack -lopenblas -o $(TARGET_CPD) $(CXXFLAGS)

$(OBJ_COMMON_CMD): %_cmd.o: %.cpp
	$(MPICXX) -I ./cpp/cmd -D MODEL_CMD -c $< -o $@ $(CXXFLAGS)

$(OBJ_COMMON_RENYI): %_renyi.o: %.cpp
	$(MPICXX) -I ./cpp/renyi -D MODEL_RENYI -c $< -o $@ $(CXXFLAGS)

$(OBJ_COMMON_CPD): %_cpd.o: %.cpp
	$(MPICXX) -I ./cpp/cpd -D MODEL_CPD -c $< -o $@ $(CXXFLAGS)

$(OBJ_OLD_OLD): %.o: %.cpp
	$(MPICXX) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_OLD): %.o: %.cpp
	$(MPICXX) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_CMD): %.o: %.cpp
	$(MPICXX) -I ./cpp/cmd -c $< -o $@ $(CXXFLAGS)

$(OBJ_RENYI): %.o: %.cpp
	$(MPICXX) -I ./cpp/renyi -c $< -o $@ $(CXXFLAGS)

$(OBJ_CPD): %.o: %.cpp
	$(MPICXX) -I ./cpp/cpd -c $< -o $@ $(CXXFLAGS)

clean:
	# @rm -f ./cpp_old_old/*.o 2>/dev/null || true
	# @rm -f ./cpp_old/*.o 2>/dev/null || true
	# @rm $(TARGET_OLD_OLD) 2>/dev/null || true
	# @rm $(TARGET_OLD) 2>/dev/null || true
	@rm -f ./cpp/*.o 2>/dev/null || true
	@rm -f ./cpp/*/*.o 2>/dev/null || true
	@rm $(TARGET_CMD) 2>/dev/null || true
	@rm $(TARGET_RENYI) 2>/dev/null || true
	@rm $(TARGET_CPD) 2>/dev/null || true

graph: 
	@make -dn MAKE=: all | sed -rn "s/^(\s+)Considering target file '(.*)'\.$$/\1\2/p"