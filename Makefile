# SRC = $(wildcard *.c)
# OBJ = $(SRC:.c=.o)

# all: $(OBJ)
# 	gcc -shared -o final.so $(OBJ)

# %.o: %.c
# 	gcc -c -fPIC $< -o $@

# clean:
# 	rm -f *.o final.so


# CC = gcc  # Change this if you use a different compiler
# CFLAGS = -O3 -Wall -shared -fPIC  # Flags for warnings, shared library, PIC

# # Target to create the shared library
# all: final.so

# # Rule to compile and link the shared library
# final.so: BigInt.o
# 	$(CC) $(CFLAGS) -o final.so uint.o

# # Rule to compile the C source code
# uint.o: uint.c
# 	$(CC) $(CFLAGS) -c uint.c

# # Rule to clean up (remove object file and shared library)
# clean:
# 	rm -f uint.so *.o

# CC = gcc
# CFLAGS = -O3 -Wall -shared -fPIC  # Optimization, warnings, shared library, PIC

# SRC = $(wildcard *.c)
# OBJ = $(SRC:.c=.o)

# all: $(OBJ)
# 	$(CC) $(CFLAGS) -o final.so $(OBJ)

# %.o: %.c
# 	$(CC) $(CFLAGS) -c $< -o $@

# clean:
# 	rm -f *.o final.so

CC = gcc
CFLAGS = -O3 -Wall -shared -fPIC

SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)

all: $(OBJ)
	$(CC) $(CFLAGS) -o final.so $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o final.so