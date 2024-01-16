MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -ffloat-store -ffp-contract=off -march=native -pipe -std=c++17
# CXXFLAGS = -O0

TARGET_OLD = ./bin/tree_approx_potts_old
TARGET = ./bin/tree_approx_potts
TARGET_CMD = ./bin/cmd_approx
TARGET_RENYI = ./bin/renyi_approx
SRC_OLD = ./cpp_old/tree_approx_potts.cpp ./cpp_old/algorithm.cpp ./cpp_old/graph.cpp ./cpp_old/graph_utils.cpp ./cpp_old/site.cpp ./cpp_old/bond.cpp ./cpp_old/bond_utils.cpp
SRC = ./cpp/tree_approx_potts.cpp ./cpp/observables.cpp ./cpp/algorithm.cpp ./cpp/graph_utils.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/bond_utils.cpp ./cpp/mpi_utils.cpp
SRC_COMMON = ./cpp_cmd/main.cpp ./cpp_cmd/observables.cpp ./cpp_cmd/site.cpp ./cpp_cmd/mpi_utils.cpp
SRC_CMD = ./cpp_cmd/cmd/graph_utils.cpp ./cpp_cmd/cmd/algorithm.cpp ./cpp_cmd/cmd/optimize.cpp ./cpp_cmd/cmd/bond.cpp
SRC_RENYI = ./cpp_cmd/renyi/graph_utils.cpp ./cpp_cmd/renyi/algorithm.cpp ./cpp_cmd/renyi/optimize.cpp ./cpp_cmd/renyi/bond.cpp
OBJ_OLD = $(SRC_OLD:%.cpp=%.o)
OBJ = $(SRC:%.cpp=%.o)
OBJ_COMMON = $(SRC_COMMON:%.cpp=%.o)
OBJ_CMD = $(SRC_CMD:%.cpp=%.o)
OBJ_RENYI = $(SRC_RENYI:%.cpp=%.o)

.PHONY: clean
.SUFFIXES: .cpp .hpp .o

all: $(TARGET_CMD) $(TARGET_RENYI)
# all: $(TARGET) $(TARGET_CMD) $(TARGET_RENYI)
# all: $(TARGET) $(TARGET_OLD) $(TARGET_CMD) $(TARGET_RENYI)

$(TARGET_OLD): $(OBJ_OLD)
	$(CXX) $(OBJ_OLD) -o $(TARGET_OLD) $(CXXFLAGS)
	
$(TARGET): $(OBJ)
	$(MPICXX) $(OBJ) -o $(TARGET) $(CXXFLAGS)

$(TARGET_CMD): INCLUDE_PATH = ./cpp_cmd/cmd
$(TARGET_CMD): $(OBJ_COMMON) $(OBJ_CMD)
	$(MPICXX) $(OBJ_COMMON) $(OBJ_CMD) -o $(TARGET_CMD) $(CXXFLAGS)

$(TARGET_RENYI): INCLUDE_PATH = ./cpp_cmd/renyi
$(TARGET_RENYI): $(OBJ_COMMON) $(OBJ_RENYI)
	$(MPICXX) $(OBJ_COMMON) $(OBJ_RENYI) -o $(TARGET_RENYI) $(CXXFLAGS)

$(OBJ_OLD): $(@:%.o=%.cpp)
	$(MPICXX) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ): $(@:%.o=%.cpp)
	$(MPICXX) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_COMMON): $(@:%.o=%.cpp)
	$(MPICXX) -I $(INCLUDE_PATH) -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_CMD): $(@:%.o=%.cpp)
	$(MPICXX) -I ./cpp_cmd/cmd -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

$(OBJ_RENYI): $(@:%.o=%.cpp)
	$(MPICXX) -I ./cpp_cmd/renyi -c $(@:%.o=%.cpp) -o $@ $(CXXFLAGS)

clean:
	# @rm -f ./cpp_old/*.o 2>/dev/null || true
	# @rm -f ./cpp/*.o 2>/dev/null || true
	@rm -f ./cpp_cmd/*.o 2>/dev/null || true
	@rm -f ./cpp_cmd/*/*.o 2>/dev/null || true
	# @rm $(TARGET_OLD) 2>/dev/null || true
	# @rm $(TARGET) 2>/dev/null || true
	# @rm $(TARGET_CMD) 2>/dev/null || true
	@rm $(TARGET_RENYI) 2>/dev/null || true