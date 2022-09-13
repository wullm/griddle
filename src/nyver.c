/*******************************************************************************
 * This file is part of Nyver.
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

#include "../include/nyver.h"

int main(int argc, char *argv[]) {
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
        message(rank, "    | \\ | | `  ` __   __ ___  _ __ \n");
        message(rank, "    |  \\| || || |\\ \\ / // _ \\| '__|    \n");
        message(rank, "    | |\\  ||_|| | \\ V /|  __/| |   \n");
        message(rank, "    |_| \\_|   / |  \\_/  \\___||_|   \n");
        message(rank, "            |__/                   \n");
        message(rank, "\n");
        if (argc == 1) {
            printf("No parameter file specified.\n");
            return 0;
        } else {
            message(rank, "The parameter file is '%s'.\n", fname);
        }
    }

    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);

    /* Main nyver structuress */
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

    /* Integration limits */
    const double a_begin = pars.ScaleFactorBegin;
    const double a_end = pars.ScaleFactorEnd;
    const double a_factor = 1.0 + pars.ScaleFactorStep;
    const double z_start = 1.0 / a_begin - 1.0;
    const double D_start = strooklat_interp(&spline_z, ptdat.D_growth, z_start);

    /* The 2LPT factor */
    const double factor_2lpt = 3. / 7.;
    const double factor_vel_2lpt = factor_2lpt * 2.0;

    /* Allocate distributed memory arrays (one complex & one real) */
    alloc_local_grid(&lpt_potential, N, boxlen, MPI_COMM_WORLD);

    /* Generate LPT potential grid */
    generate_potential_grid(&lpt_potential, &seed, pars.FixedModes,
                            pars.InvertedModes, &ptdat, &cosmo, z_start);

    /* Allocate additional arrays */
    struct distributed_grid temp2;
    struct distributed_grid temp1;
    struct distributed_grid lpt_potential_2;
    alloc_local_grid(&temp1, N, boxlen, MPI_COMM_WORLD);
    alloc_local_grid(&temp2, N, boxlen, MPI_COMM_WORLD);
    alloc_local_grid(&lpt_potential_2, N, boxlen, MPI_COMM_WORLD);

    /* Generate the 2LPT potential grid */
    generate_2lpt_grid(&lpt_potential, &temp1, &temp2, &lpt_potential_2, &ptdat,
                       &cosmo, z_start);

    /* Free working memory used in the 2LPT calculation */
    free_local_grid(&temp1);
    free_local_grid(&temp2);

    /* Allocate memory for a particle lattice */
    struct particle *particles = malloc(N * N * N * sizeof(struct particle));

    /* Generate a particle lattice */
    generate_particle_lattice(&lpt_potential, &lpt_potential_2, &ptdat, &ptpars,
                              particles, &cosmo, &us, &pcs, z_start);


    /* Set velocities to zero when running with COLA */
    if (pars.WithCOLA) {
        for (long long i = 0; i < N*N*N; i++) {
            struct particle *p = &particles[i];

            p->v[0] = 0.;
            p->v[1] = 0.;
            p->v[2] = 0.;
        }
    }

    /* Allocate distributed memory arrays (one complex & one real) */
    struct distributed_grid mass;
    alloc_local_grid(&mass, N, boxlen, MPI_COMM_WORLD);
    mass.momentum_space = 0;

    /* First mass deposition */
    mass_deposition(&mass, particles);

    /* Compute the gravitational potential */
    compute_potential(&mass, &pcs);

    /* Start at the beginning */
    double a = a_begin;

    /* Prepare integration */
    int MAX_ITER = (log(a_end) - log(a_begin))/log(a_factor) + 1;

    /* The main loop */
    for (int ITER = 0; ITER < MAX_ITER; ITER++) {

        /* Determine the next scale factor */
        double a_next;
        if (ITER == 0) {
            a_next = a; //start with a step that does nothing
        } else if (ITER < MAX_ITER - 1) {
            a_next = a * a_factor;
        } else {
            a_next = a_end;
        }

        /* Compute the current redshift and log conformal time */
        double z = 1./a - 1.;
        double log_tau = strooklat_interp(&spline_z, ptdat.log_tau, z);

        /* Determine the half-step scale factor */
        double a_half = sqrt(a_next * a);

        /* Find the next and half-step conformal times */
        double z_next = 1./a_next - 1.;
        double z_half = 1./a_half - 1.;
        double log_tau_next = strooklat_interp(&spline_z, ptdat.log_tau, z_next);
        double log_tau_half = strooklat_interp(&spline_z, ptdat.log_tau, z_half);
        double dtau1 = exp(log_tau_half) - exp(log_tau);
        double dtau2 = exp(log_tau_next) - exp(log_tau_half);
        double dtau = dtau1 + dtau2;

        /* Obtain the growth factors */
        double D = strooklat_interp(&spline_z, ptdat.D_growth, z);
        double D_half = strooklat_interp(&spline_z, ptdat.D_growth, z_half);
        double D_next = strooklat_interp(&spline_z, ptdat.D_growth, z_next);

        /* Obtain the velocity factor derivatives */
        double d_vf = 0., d_vf_2 = 0., d_vf_half = 0., d_vf_2_half = 0.;
        if (pars.WithCOLA) {
            double f = strooklat_interp(&spline_z, ptdat.f_growth, z);
            double f_half = strooklat_interp(&spline_z, ptdat.f_growth, z_half);
            double H = strooklat_interp(&spline_z, ptdat.Hubble_H, z);
            double H_half = strooklat_interp(&spline_z, ptdat.Hubble_H, z_half);
            double vel_fact = a * a * f * H;
            double vel_fact_half = a_half * a_half * f_half * H_half;

            /* Differentiate the velocity factor at time a */
            double a_bit = a * 1.01;
            double z_bit = 1./a_bit - 1.;
            double log_tau_bit = strooklat_interp(&spline_z, ptdat.log_tau, z_bit);
            double tau_bit = exp(log_tau_bit);

            double f_bit = strooklat_interp(&spline_z, ptdat.f_growth, z_bit);
            double D_bit = strooklat_interp(&spline_z, ptdat.D_growth, z_bit);
            double H_bit = strooklat_interp(&spline_z, ptdat.Hubble_H, z_bit);
            double vel_fact_bit = a_bit * a_bit * f_bit * H_bit;

            /* Derivative of the first- and second-order velocity factors */
            d_vf = (vel_fact_bit * D_bit - vel_fact * D) / (tau_bit - exp(log_tau));
            d_vf_2 = (vel_fact_bit * D_bit * D_bit - vel_fact * D * D) / (tau_bit - exp(log_tau));

            /* Differentiate the velocity factor at time a_half */
            double a_hbit = a_half * 1.01;
            double z_hbit = 1./a_hbit - 1.;
            double log_tau_hbit = strooklat_interp(&spline_z, ptdat.log_tau, z_hbit);
            double tau_hbit = exp(log_tau_hbit);

            double f_hbit = strooklat_interp(&spline_z, ptdat.f_growth, z_hbit);
            double D_hbit = strooklat_interp(&spline_z, ptdat.D_growth, z_hbit);
            double H_hbit = strooklat_interp(&spline_z, ptdat.Hubble_H, z_hbit);
            double vel_fact_hbit = a_hbit * a_hbit * f_hbit * H_hbit;

            /* Derivative of the first- and second-order velocity factors */
            d_vf_half = (vel_fact_hbit * D_hbit - vel_fact_half * D_half) / (tau_hbit - exp(log_tau_half));
            d_vf_2_half = (vel_fact_hbit * D_hbit * D_hbit - vel_fact_half * D_half * D_half) / (tau_hbit - exp(log_tau_half));
        }

        /* Skip the particle integration during the first step */
        if (ITER == 0)
            continue;

        /* Integrate the particles */
        for (long long i = 0; i < N*N*N; i++) {
            struct particle *p = &particles[i];

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            accelCIC(&mass, N, boxlen, p->x, acc);

            /* Execute first half-kick */
            p->v[0] += acc[0] * dtau1;
            p->v[1] += acc[1] * dtau1;
            p->v[2] += acc[2] * dtau1;

            /* COLA half-kick */
            if (pars.WithCOLA) {
                p->v[0] += d_vf * dtau1 * p->dx[0] / D_start - d_vf_2 * dtau1 * p->dx2[0] / (D_start * D_start) * factor_vel_2lpt;
                p->v[1] += d_vf * dtau1 * p->dx[1] / D_start - d_vf_2 * dtau1 * p->dx2[1] / (D_start * D_start) * factor_vel_2lpt;
                p->v[2] += d_vf * dtau1 * p->dx[2] / D_start - d_vf_2 * dtau1 * p->dx2[2] / (D_start * D_start) * factor_vel_2lpt;
            }

            /* Execute drift (only one drift, so use dtau = dtau1 + dtau2) */
            p->x[0] += p->v[0] * dtau / a_half;
            p->x[1] += p->v[1] * dtau / a_half;
            p->x[2] += p->v[2] * dtau / a_half;

            /* COLA drift */
            if (pars.WithCOLA) {
                p->x[0] -= p->dx[0] * (D_next - D) / D_start - factor_2lpt * p->dx2[0] * (D_next * D_next - D * D) / (D_start * D_start);
                p->x[1] -= p->dx[1] * (D_next - D) / D_start - factor_2lpt * p->dx2[1] * (D_next * D_next - D * D) / (D_start * D_start);
                p->x[2] -= p->dx[2] * (D_next - D) / D_start - factor_2lpt * p->dx2[2] * (D_next * D_next - D * D) / (D_start * D_start);
            }
        }

        /* Initiate mass deposition */
        mass_deposition(&mass, particles);

        /* Re-compute the gravitational potential */
        compute_potential(&mass, &pcs);

        /* Integrate the particles */
        for (long long i = 0; i < N*N*N; i++) {
            struct particle *p = &particles[i];

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            accelCIC(&mass, N, boxlen, p->x, acc);

            /* Execute second half-kick */
            p->v[0] += acc[0] * dtau2;
            p->v[1] += acc[1] * dtau2;
            p->v[2] += acc[2] * dtau2;

            /* COLA half-kick */
            if (pars.WithCOLA) {
                p->v[0] += d_vf_half * dtau2 * p->dx[0] / D_start - d_vf_2_half * dtau2 * p->dx2[0] / (D_start * D_start) * factor_vel_2lpt;
                p->v[1] += d_vf_half * dtau2 * p->dx[1] / D_start - d_vf_2_half * dtau2 * p->dx2[1] / (D_start * D_start) * factor_vel_2lpt;
                p->v[2] += d_vf_half * dtau2 * p->dx[2] / D_start - d_vf_2_half * dtau2 * p->dx2[2] / (D_start * D_start) * factor_vel_2lpt;
            }
        }

        /* Step forward */
        a = a_next;

        printf("%d %g %g\n", ITER, a, z);
    }

    /* Initiate mass deposition */
    mass_deposition(&mass, particles);

    /* Export the GRF */
    writeFieldFile_dg(&mass, "mass.hdf5");

    /* Export a snapshot */
    exportSnapshot(&pars, &us, particles, "snap.hdf5", N);

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
