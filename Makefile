MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -ffloat-store -ffp-contract=off -march=native -pipe -std=c++17
# CXXFLAGS = -O0

TARGET_OLD_OLD = ./bin/tree_approx_potts_old_OLD
TARGET_OLD = ./bin/tree_approx_potts_OLD
TARGET_CMD = ./bin/cmd_approx
TARGET_RENYI = ./bin/renyi_approx
SRC_OLD_OLD = ./cpp_old_old/tree_approx_potts.cpp ./cpp_old_old/algorithm.cpp ./cpp_old_old/graph.cpp ./cpp_old_old/graph_utils.cpp ./cpp_old_old/site.cpp ./cpp_old_old/bond.cpp ./cpp_old_old/bond_utils.cpp
SRC_OLD = ./cpp_old/tree_approx_potts.cpp ./cpp_old/observables.cpp ./cpp_old/algorithm.cpp ./cpp_old/graph_utils.cpp ./cpp_old/site.cpp ./cpp_old/bond.cpp ./cpp_old/bond_utils.cpp ./cpp_old/mpi_utils.cpp
SRC_COMMON = ./cpp/main.cpp ./cpp/observables.cpp ./cpp/site.cpp ./cpp/mpi_utils.cpp
SRC_CMD = ./cpp/cmd/graph_utils.cpp ./cpp/cmd/algorithm.cpp ./cpp/cmd/optimize.cpp ./cpp/cmd/bond.cpp
SRC_RENYI = ./cpp/renyi/graph_utils.cpp ./cpp/renyi/algorithm.cpp ./cpp/renyi/optimize.cpp ./cpp/renyi/bond.cpp
OBJ_OLD_OLD = $(SRC_OLD_OLD:%.cpp=%.o)
OBJ_OLD = $(SRC_OLD:%.cpp=%.o)
OBJ_COMMON = $(SRC_COMMON:%.cpp=%.o)
OBJ_CMD = $(SRC_CMD:%.cpp=%.o)
OBJ_RENYI = $(SRC_RENYI:%.cpp=%.o)

.PHONY: clean
.SUFFIXES: .cpp .hpp .o

all: $(TARGET_CMD) $(TARGET_RENYI)
# all: $(TARGET_OLD) $(TARGET_CMD) $(TARGET_RENYI)
# all: $(TARGET_OLD_OLD) $(TARGET_OLD) $(TARGET_CMD) $(TARGET_RENYI)

$(TARGET_OLD_OLD): $(OBJ_OLD_OLD)
	$(CXX) $(OBJ_OLD_OLD) -o $(TARGET_OLD_OLD) $(CXXFLAGS)
	
$(TARGET_OLD): $(OBJ_OLD)
	$(MPICXX) $(OBJ_OLD) -o $(TARGET_OLD) $(CXXFLAGS)

$(TARGET_CMD): INCLUDE_PATH = ./cpp/cmd
$(TARGET_CMD): $(OBJ_COMMON) $(OBJ_CMD)
	$(MPICXX) $(OBJ_COMMON) $(OBJ_CMD) -o $(TARGET_CMD) $(CXXFLAGS)

$(TARGET_RENYI): INCLUDE_PATH = ./cpp/renyi
$(TARGET_RENYI): $(OBJ_COMMON) $(OBJ_RENYI)
	$(MPICXX) $(OBJ_COMMON) $(OBJ_RENYI) -o $(TARGET_RENYI) $(CXXFLAGS)

$(OBJ_OLD_OLD): $(@:%.o=%.cpp)
	$(MPICXX) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_OLD): $(@:%.o=%.cpp)
	$(MPICXX) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_COMMON): $(@:%.o=%.cpp)
	$(MPICXX) -I $(INCLUDE_PATH) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_CMD): $(@:%.o=%.cpp)
	$(MPICXX) -I ./cpp/cmd -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_RENYI): $(@:%.o=%.cpp)
	$(MPICXX) -I ./cpp/renyi -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

clean:
	# @rm -f ./cpp_old_old/*.o 2>/dev/null || true
	# @rm -f ./cpp_old/*.o 2>/dev/null || true
	# @rm $(TARGET_OLD_OLD) 2>/dev/null || true
	# @rm $(TARGET_OLD) 2>/dev/null || true
	@rm -f ./cpp/*.o 2>/dev/null || true
	@rm -f ./cpp/*/*.o 2>/dev/null || true
	@rm $(TARGET_CMD) 2>/dev/null || true
	@rm $(TARGET_RENYI) 2>/dev/null || true