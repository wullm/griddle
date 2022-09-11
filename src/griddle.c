/*******************************************************************************
 * This file is part of griddle.
 * Copyright (c) 2022 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <complex.h>
#include <mpi.h>
#include <fftw3-mpi.h>

#include "../include/griddle.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Read options */
    const char *fname = argv[1];
    if (rank == 0) {
        message(rank, "\n");
        header(rank, "Welcome to griddle; please wait a little.\n");
        message(rank, "The parameter file is '%s'\n", fname);
    }

    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);
    
    /* Main griddle structuress */
    struct params pars;
    struct units us;
    struct physical_consts pcs;
    struct cosmology cosmo;
    
    /* Structures for dealing with perturbation data (transfer functions) */
    struct perturb_data ptdat;
    struct perturb_params ptpars;
        
    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    set_physical_constants(&us, &pcs);
    readCosmology(&cosmo, fname);
        
    /* Read the perturbation data file */
    readPerturb(&us, &ptdat, pars.TransferFunctionsFile);
    readPerturbParams(&us, &ptpars, pars.TransferFunctionsFile);
        
    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat.redshift, ptdat.tau_size};
    struct strooklat spline_k = {ptdat.k, ptdat.k_size};
    init_strooklat_spline(&spline_z, 100);
    init_strooklat_spline(&spline_k, 100);
        
    /* Store the MPI rank */
    pars.rank = rank;
    
    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed + rank);
    
    /* Create or read a Gaussian random field */
    int N = pars.GridSize;
    double boxlen = pars.BoxLength;
    struct distributed_grid lpt_potential;
    
    /* The velocity pre-factor a * H * f */
    const double z_start = pars.z_start;
    
        
    /* Allocate distributed memory arrays (one complex & one real) */
    alloc_local_grid(&lpt_potential, N, boxlen, MPI_COMM_WORLD);
    
    /* Generate LPT potential grid */
    generate_potential_grid(&lpt_potential, &seed, &ptdat, &cosmo, z_start);
    
    /* Allocate memory for a particle lattice */
    struct particle *particles = malloc(N * N * N * sizeof(struct particle));
    
    /* Generate a particle lattice */
    generate_particle_lattice(&lpt_potential, &ptdat, particles, z_start);
    
    
        
    /* Allocate distributed memory arrays (one complex & one real) */
    struct distributed_grid mass;
    alloc_local_grid(&mass, N, boxlen, MPI_COMM_WORLD);
    mass.momentum_space = 0;
    
    mass_deposition(&mass, particles);
    
    /* Export the GRF */
    writeFieldFile_dg(&mass, "mass.hdf5");
    
    
    
    /* Free the particle lattice */
    free(particles);   
                                
    /* Free the LPT potential grids */
    free_local_grid(&lpt_potential);
    
    /* Free the mass grid */
    free_local_grid(&mass);
    
    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanParams(&pars);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);
    
    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    free_strooklat_spline(&spline_k);
    
    /* Timer */
    gettimeofday(&time_stop, NULL);
    long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
                           + time_stop.tv_usec - time_start.tv_usec;
    message(rank, "\nTime elapsed: %.5f s\n", microsec/1e6);

}
