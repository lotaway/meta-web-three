CC = GCC
OBJ := $(shell cd /app/src \ ls *.cpp)
INCLUDES = -l./include
target = main
$(target) : $(OBJ)
	$(CC) $(INCLUDES) $(OBJ) -o $^ $@

.PHONY: clean
clean:
	-rm ./main
	-#rm -rf ${OBJ}