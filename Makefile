# Compiler
NVCC = nvcc

# Flags:
# -O3: Optimization
# -I./include: Header path
# -lcurand: Random number library
# -lnvToolsExt: Profiler library
# Note: We removed -std=c++17 to let NVCC use its default compatibility mode
NVCC_FLAGS = -O3 -I./include -lcurand -lnvToolsExt

# Directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data

# Target
TARGET = $(BIN_DIR)/risk_engine

# Sources
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Rules
all: clean $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	@echo "Compiling Safe Version..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(BIN_DIR)/* $(DATA_DIR)/*.csv $(DATA_DIR)/*.png