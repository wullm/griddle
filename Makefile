#Compiler options
GCC = mpicc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3 -lfftw3f -lfftw3_omp -lfftw3_mpi
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -I/usr/include/hdf5/openmpi

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES) $(FIREBOLT_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES) $(FIREBOLT_LIBRARIES)
CFLAGS = -Wall -Wshadow=global -fopenmp -march=native -O4
LDFLAGS =

OBJECTS = lib/*.o

all:
	make minIni
	mkdir -p lib
	$(GCC) src/params.c -c -o lib/params.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/units.c -c -o lib/units.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/grid_io.c -c -o lib/grid_io.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/random.c -c -o lib/random.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/distributed_grid.c -c -o lib/distributed_grid.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/fft.c -c -o lib/fft.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/gaussian_field.c -c -o lib/gaussian_field.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/perturb_data.c -c -o lib/perturb_data.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/cosmology.c -c -o lib/cosmology.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/cosmology_tables.c -c -o lib/cosmology_tables.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/initial_conditions.c -c -o lib/initial_conditions.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/mesh_grav.c -c -o lib/mesh_grav.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/mass_deposit.c -c -o lib/mass_deposit.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/snap_io.c -c -o lib/snap_io.o $(INCLUDES) $(CFLAGS)

	$(GCC) src/nyver.c -o nyver $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

minIni:
	cd parser && make
