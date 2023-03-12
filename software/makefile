CC = GCC
OBJ := $(shell cd /app/src \ ls *.cpp)
include = -l./include
target = main
$(target) : $(OBJ)
	$(CC) $(OBJ) -o $^ $@

.PHONY: clean
clean:
	-rm main