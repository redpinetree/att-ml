MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -ffloat-store -ffp-contract=off -march=native -pipe -std=c++17
# CXXFLAGS = -O0

TARGET = ./bin/tree_approx_potts
TARGET_CMD = ./bin/cmd_approx
TARGET_OLD = ./bin/tree_approx_potts_old
SRC = ./cpp/tree_approx_potts.cpp ./cpp/observables.cpp ./cpp/algorithm.cpp ./cpp/graph_utils.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/bond_utils.cpp ./cpp/mpi_utils.cpp
SRC_CMD = ./cpp_cmd/cmd_approx.cpp ./cpp_cmd/observables.cpp ./cpp_cmd/algorithm.cpp ./cpp_cmd/optimize.cpp ./cpp_cmd/graph_utils.cpp ./cpp_cmd/site.cpp ./cpp_cmd/bond.cpp ./cpp_cmd/mpi_utils.cpp
SRC_OLD = ./cpp_old/tree_approx_potts.cpp ./cpp_old/algorithm.cpp ./cpp_old/graph.cpp ./cpp_old/graph_utils.cpp ./cpp_old/site.cpp ./cpp_old/bond.cpp ./cpp_old/bond_utils.cpp
OBJ = $(SRC:%.cpp=%.o)
OBJ_CMD = $(SRC_CMD:%.cpp=%.o)
OBJ_OLD = $(SRC_OLD:%.cpp=%.o)

.PHONY: clean
.SUFFIXES: .cpp .hpp .o

all: $(TARGET_CMD)
all: $(TARGET) $(TARGET_CMD)
all: $(TARGET) $(TARGET_OLD) $(TARGET_CMD)

$(TARGET): $(OBJ)
	$(MPICXX) $(OBJ) -o $(TARGET) $(CXXFLAGS)

$(TARGET_CMD): $(OBJ_CMD)
	$(MPICXX) $(OBJ_CMD) -o $(TARGET_CMD) $(CXXFLAGS)

$(TARGET_OLD): $(OBJ_OLD)
	$(CXX) $(OBJ_OLD) -o $(TARGET_OLD) $(CXXFLAGS)

%.o: %.cpp
	$(MPICXX) -c $< -o $@ $(CXXFLAGS)

clean:
	#@rm -f ./cpp/*.o 2>/dev/null || true
	@rm -f ./cpp_cmd/*.o 2>/dev/null || true
	#@rm -f ./cpp_old/*.o 2>/dev/null || true
	#@rm $(TARGET) 2>/dev/null || true
	@rm $(TARGET_CMD) 2>/dev/null || true
	#@rm $(TARGET_OLD) 2>/dev/null || true