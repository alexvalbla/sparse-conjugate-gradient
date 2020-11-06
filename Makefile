# your choice of compiler
CC = gcc-8

# Add your choice of flags
CFLAGS = -O3 -Wall -Wextra -g -fopenmp
LDLIBS = -lm

##############################################

all : cg cg_omp cg_omp2

#object files
mmio.o : mmio.c mmio.h

cg.o : cg.c mmio.h
	gcc-8 -Wall -Wextra -O3 -g -c -o cg.o cg.c

cg_omp.o : cg_omp.c mmio.h
	gcc-8 -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -c -o cg_omp.o cg_omp.c

cg_omp2.o : cg_omp2.c mmio.h
	gcc-8 -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -c -o cg_omp2.o cg_omp2.c

cg_omp_avx512.o : cg_omp_avx512.c mmio.h
	gcc-8 -Wall -Wextra -O3 -g -fopenmp -mavx512f -mfma -c -o cg_omp_avx512.o cg_omp_avx512.c

cg_mpi.o : cg_mpi.c mmio.h
	mpicc -Wall -Wextra -O3 -g -mavx2 -mfma -c -o cg_mpi.o cg_mpi.c -lm

cg_mpi_omp.o : cg_mpi_omp.c mmio.h
	mpicc -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -c -o cg_mpi_omp.o cg_mpi_omp.c -lm

cg_mpi_omp2.o : cg_mpi_omp2.c mmio.h
	mpicc -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -c -o cg_mpi_omp2.o cg_mpi_omp2.c -lm

#executables
cg : cg.o mmio.o

cg_omp : cg_omp.o mmio.o
	gcc-8 -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -o cg_omp cg_omp.o mmio.o

cg_omp2 : cg_omp2.o mmio.o
	gcc-8 -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -o cg_omp2 cg_omp2.o mmio.o

cg_omp_avx512 : cg_omp_avx512.o mmio.o
	gcc-8 -Wall -Wextra -O3 -g -fopenmp -mavx512f -mfma -o cg_omp_avx512 cg_omp_avx512.o mmio.o

cg_mpi : cg_mpi.o mmio.o
	mpicc -Wall -Wextra -O3 -g -o -mavx2 -mfma cg_mpi cg_mpi.o mmio.o -lm

cg_mpi_omp : cg_mpi_omp.o mmio.o
	mpicc -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -o cg_mpi_omp cg_mpi_omp.o mmio.o -lm

cg_mpi_omp2 : cg_mpi_omp2.o mmio.o
	mpicc -Wall -Wextra -O3 -g -fopenmp -mavx2 -mfma -o cg_mpi_omp2 cg_mpi_omp2.o mmio.o -lm


.PHONY: clean
clean :
	rm -rf *.o cg cg_omp cg_omp2
