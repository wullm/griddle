#Compiler options
GCC = mpicc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3 -lfftw3f -lfftw3_omp -lfftw3_mpi -lfftw3f_mpi
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

# The main sources list
SOURCES =
SOURCES += params units grid_io random fermi_dirac distributed_grid particle_exchange
SOURCES += fft gaussian_field perturb_data cosmology initial_conditions mesh_grav
SOURCES += mass_deposit snap_io analysis_fof analysis_so catalogue_io

# The corresponding objects
OBJECTS = $(patsubst %, lib/%.o, $(SOURCES))

# The corresponding dependencies
DEPENDS = $(patsubst %, lib/%.d, $(SOURCES))

all: minIni lib
	./git_version.sh
	make sedulus

sedulus: $(OBJECTS) src/sedulus.c include/git_version.h
	$(GCC) src/sedulus.c -o sedulus $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

-include $(DEPENDS)

$(OBJECTS) : lib/%.o : src/%.c
	$(GCC) $< -c -MMD -o $@ $(INCLUDES) $(CFLAGS)

clean:
	rm -f lib/*.o
	rm -f lib/*.d
	rm -f parser/*.o
	rm -f sedulus

minIni:
	cd parser && make

lib:
	mkdir -p lib
