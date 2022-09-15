/*******************************************************************************
 * This file is part of Sedulus.
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

#include "../include/sedulus.h"

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
        message(rank, "     ____           _       _               \n");
        message(rank, "    / ___|  ___  __| |_   _| |_   _ ___     \n");
        message(rank, "    \\___ \\ / _ \\/ _` | | | | | | | / __| \n");
        message(rank, "     ___) |  __/ (_| | |_| | | |_| \\__ \\  \n");
        message(rank, "    |____/ \\___|\\__,_|\\__,_|_|\\__,_|___/\n");
        message(rank, "\n");
        if (argc == 1) {
            printf("No parameter file specified.\n");
            return 0;
        } else {
            message(rank, "The parameter file is '%s'.\n", fname);
            message(rank, "\n");
        }
    }

    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);

    /* Main Sedulus structuress */
    struct params pars;
    struct units us;
    struct physical_consts pcs;
    struct cosmology cosmo;
    struct cosmology_tables ctabs;

    /* Structures for dealing with perturbation data (transfer functions) */
    struct perturb_data ptdat;
    struct perturb_params ptpars;

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    set_physical_constants(&us, &pcs);
    readCosmology(&cosmo, fname);

    /* Print information about the cosmological model */
    print_cosmology_information(rank, &cosmo);

    /* Integration limits */
    const double a_begin = pars.ScaleFactorBegin;
    const double a_end = pars.ScaleFactorEnd;
    const double a_factor = 1.0 + pars.ScaleFactorStep;
    const double z_start = 1.0 / a_begin - 1.0;

    /* Integrate the background cosmology */
    integrate_cosmology_tables(&cosmo, &us, &pcs, &ctabs, 1e-3, fmax(1.0, a_end) * 1.01, 1000);

    /* Read the perturbation data file */
    readPerturb(&us, &ptdat, pars.TransferFunctionsFile);
    readPerturbParams(&us, &ptpars, pars.TransferFunctionsFile);

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat.redshift, ptdat.tau_size};
    struct strooklat spline_k = {ptdat.k, ptdat.k_size};
    init_strooklat_spline(&spline_z, 100);
    init_strooklat_spline(&spline_k, 100);

    /* Additional interpolation spline for the background cosmology scale factors */
    struct strooklat spline_bg_a = {ctabs.avec, ctabs.size};
    init_strooklat_spline(&spline_bg_a, 100);

    /* Store the MPI rank */
    pars.rank = rank;

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed + rank);

    /* Create or read a Gaussian random field */
    int N = pars.GridSize;
    double boxlen = pars.BoxLength;
    struct distributed_grid lpt_potential;

    /* Check what portions of 3D grids get stored locally */
    long int X0, NX;
    fftw_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &NX, &X0);

    /* Each MPI rank stores a portion of the full 3D mesh, as well as copies
     * of the edges of its left & right neighbours, necessary for interpolation
     * and mass deposition. These are called buffers. */

    /* Check that the buffer sizes are valid */
    const int buffer_width = DEFAULT_BUFFER_WIDTH;
    if (buffer_width > N / MPI_Rank_Count) {
        printf("There are too many MPI ranks - increase the grid size!\n");
        exit(1);
    }

    /* The 2LPT factor */
    const double D_start = strooklat_interp(&spline_z, ptdat.D_growth, z_start);
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

    /* Create buffers for the LPT potentials */
    alloc_local_buffers(&lpt_potential, buffer_width);
    alloc_local_buffers(&lpt_potential_2, buffer_width);
    create_local_buffers(&lpt_potential);
    create_local_buffers(&lpt_potential_2);

    /* Allocate memory for a particle lattice */
    // long long X0 = lpt_potential.X0;
    // long long NX = lpt_potential.NX;
    long long foreign_buffer = 10; //extra memory for exchanging particles
    long long local_partnum = NX * N * N;
    long long max_partnum = (NX + foreign_buffer) * N * N;
    struct particle *particles = malloc(max_partnum * sizeof(struct particle));

    /* Generate a particle lattice */
    generate_particle_lattice(&lpt_potential, &lpt_potential_2, &ptdat, &ptpars,
                              particles, &cosmo, &us, &pcs, X0, NX, z_start);

    /* We are done with the LPT potentials */
    free_local_grid(&lpt_potential);
    free_local_grid(&lpt_potential_2);

    /* Set velocities to zero when running with COLA */
    if (pars.WithCOLA) {
        for (long long i = 0; i < local_partnum; i++) {
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

    /* Create buffers for the mass grid */
    alloc_local_buffers(&mass, buffer_width);

    /* First mass deposition */
    if (MPI_Rank_Count == 1) {
        mass_deposition_single(&mass, particles, local_partnum);
    } else {
        mass_deposition(&mass, particles, local_partnum);
    }

    /* Merge the buffers with the main grid */
    add_local_buffers(&mass);

    /* Export the GRF */
    writeFieldFile_dg(&mass, "ini_mass.hdf5");

    /* Compute the gravitational potential */
    compute_potential(&mass, &pcs);

    /* Copy buffers and communicate them to the neighbour ranks */
    create_local_buffers(&mass);

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

        /* Determine the half-step scale factor */
        double a_half = sqrt(a_next * a);

        /* Find the next and half-step conformal times */
        double z_next = 1./a_next - 1.;

        /* Obtain the kick and drift time steps */
        double kick_dtau1 = strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_half) -
                            strooklat_interp(&spline_bg_a, ctabs.kick_factors, a);
        double kick_dtau2 = strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_next) -
                            strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_half);
        double drift_dtau = strooklat_interp(&spline_bg_a, ctabs.drift_factors, a_next) -
                            strooklat_interp(&spline_bg_a, ctabs.drift_factors, a);

        /* Obtain the velocity factor derivatives */
        double D = 0., D_next = 0.;
        double d_vf = 0., d_vf_2 = 0.;
        double d_vf_next = 0., d_vf_2_next = 0.;
        if (pars.WithCOLA) {
            /* Conformal time intervals */
            double log_tau = strooklat_interp(&spline_z, ptdat.log_tau, z);
            double log_tau_next = strooklat_interp(&spline_z, ptdat.log_tau, z_next);

            /* Obtain the growth factors */
            D = strooklat_interp(&spline_z, ptdat.D_growth, z);
            D_next = strooklat_interp(&spline_z, ptdat.D_growth, z_next);

            double f = strooklat_interp(&spline_z, ptdat.f_growth, z);
            double H = strooklat_interp(&spline_z, ptdat.Hubble_H, z);
            double vel_fact = a * a * f * H;

            double f_next = strooklat_interp(&spline_z, ptdat.f_growth, z_next);
            double H_next = strooklat_interp(&spline_z, ptdat.Hubble_H, z_next);
            double vel_fact_next = a_next * a_next * f_next * H_next;

            /* Differentiate the velocity factor at time a */
            double a_bit = a * 1.0001;
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

            /* Differentiate the velocity factor at time a_next */
            double a_nbit = a_next * 1.0001;
            double z_nbit = 1./a_nbit - 1.;
            double log_tau_nbit = strooklat_interp(&spline_z, ptdat.log_tau, z_nbit);
            double tau_nbit = exp(log_tau_nbit);

            double f_nbit = strooklat_interp(&spline_z, ptdat.f_growth, z_nbit);
            double D_nbit = strooklat_interp(&spline_z, ptdat.D_growth, z_nbit);
            double H_nbit = strooklat_interp(&spline_z, ptdat.Hubble_H, z_nbit);
            double vel_fact_nbit = a_nbit * a_nbit * f_nbit * H_nbit;

            /* Derivative of the first- and second-order velocity factors */
            d_vf_next = (vel_fact_nbit * D_nbit - vel_fact_next * D_next) / (tau_nbit - exp(log_tau_next));
            d_vf_2_next = (vel_fact_nbit * D_nbit * D_nbit - vel_fact_next * D_next * D_next) / (tau_nbit - exp(log_tau_next));
        }

        /* Skip the particle integration during the first step */
        if (ITER == 0)
            continue;

        message(rank, "Step %d at z = %g\n", ITER, z);
        message(rank, "Computing particle kicks and drifts.\n");

        /* Timer */
        struct timeval time_sort_a;
        gettimeofday(&time_sort_a, NULL);

        /* Integrate the particles */
        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &particles[i];

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            if (ITER == 1) {
                if (MPI_Rank_Count == 1) {
                    accelCIC_single(&mass, N, boxlen, p->x, acc);
                } else {
                    accelCIC(&mass, N, boxlen, p->x, acc);
                }
            } else {
                acc[0] = p->a[0];
                acc[1] = p->a[1];
                acc[2] = p->a[2];
            }

            /* Execute first half-kick */
            p->v[0] += acc[0] * kick_dtau1;
            p->v[1] += acc[1] * kick_dtau1;
            p->v[2] += acc[2] * kick_dtau1;

            /* COLA half-kick */
            if (pars.WithCOLA) {
                p->v[0] += d_vf * kick_dtau1 * p->dx[0] / D_start + d_vf_2 * kick_dtau1 * p->dx2[0] / (D_start * D_start) * factor_vel_2lpt;
                p->v[1] += d_vf * kick_dtau1 * p->dx[1] / D_start + d_vf_2 * kick_dtau1 * p->dx2[1] / (D_start * D_start) * factor_vel_2lpt;
                p->v[2] += d_vf * kick_dtau1 * p->dx[2] / D_start + d_vf_2 * kick_dtau1 * p->dx2[2] / (D_start * D_start) * factor_vel_2lpt;
            }

            /* Execute drift (only one drift, so use dtau = dtau1 + dtau2) */
            p->x[0] += p->v[0] * drift_dtau;
            p->x[1] += p->v[1] * drift_dtau;
            p->x[2] += p->v[2] * drift_dtau;

            /* COLA drift */
            if (pars.WithCOLA) {
                p->x[0] -= p->dx[0] * (D_next - D) / D_start + factor_2lpt * p->dx2[0] * (D_next * D_next - D * D) / (D_start * D_start);
                p->x[1] -= p->dx[1] * (D_next - D) / D_start + factor_2lpt * p->dx2[1] * (D_next * D_next - D * D) / (D_start * D_start);
                p->x[2] -= p->dx[2] * (D_next - D) / D_start + factor_2lpt * p->dx2[2] * (D_next * D_next - D * D) / (D_start * D_start);
            }

            /* Wrap particles in the periodic domain */
            p->x[0] = fwrap(p->x[0], boxlen);
            p->x[1] = fwrap(p->x[1], boxlen);
            p->x[2] = fwrap(p->x[2], boxlen);
        }

        /* Timer */
        struct timeval time_sort_b;
        gettimeofday(&time_sort_b, NULL);
        message(rank, "Evolving particles took %.5f s\n",
                               ((time_sort_b.tv_sec - time_sort_a.tv_sec) * 1000000
                               + time_sort_b.tv_usec - time_sort_a.tv_usec)/1e6);

        if (MPI_Rank_Count > 1) {

            message(rank, "Starting particle exchange.\n");

            /* Timer */
            struct timeval time_sort_0;
            gettimeofday(&time_sort_0, NULL);

            exchange_particles(particles, boxlen, &local_partnum);

            /* Timer */
            struct timeval time_sort_1;
            gettimeofday(&time_sort_1, NULL);
            message(rank, "Exchanging particles took %.5f s\n",
                                   ((time_sort_1.tv_sec - time_sort_0.tv_sec) * 1000000
                                   + time_sort_1.tv_usec - time_sort_0.tv_usec)/1e6);
        }

        message(rank, "Computing the gravitational potential.\n");

        /* Timer */
        struct timeval time_sort_1;
        gettimeofday(&time_sort_1, NULL);

        /* Initiate mass deposition */
        if (MPI_Rank_Count == 1) {
            mass_deposition_single(&mass, particles, local_partnum);
        } else {
            mass_deposition(&mass, particles, local_partnum);
        }

        /* Timer */
        struct timeval time_sort_2;
        gettimeofday(&time_sort_2, NULL);
        message(rank, "Computing mass density took %.5f s\n",
                               ((time_sort_2.tv_sec - time_sort_1.tv_sec) * 1000000
                               + time_sort_2.tv_usec - time_sort_1.tv_usec)/1e6);

        /* Merge the buffers with the main grid */
        add_local_buffers(&mass);

        /* Timer */
        struct timeval time_sort_3;
        gettimeofday(&time_sort_3, NULL);
        message(rank, "Communicating buffers took %.5f s\n",
                               ((time_sort_3.tv_sec - time_sort_2.tv_sec) * 1000000
                               + time_sort_3.tv_usec - time_sort_2.tv_usec)/1e6);

        /* Re-compute the gravitational potential */
        compute_potential(&mass, &pcs);

        /* Timer */
        struct timeval time_sort_4;
        gettimeofday(&time_sort_4, NULL);
        message(rank, "Computing the potential took %.5f s\n",
                               ((time_sort_4.tv_sec - time_sort_3.tv_sec) * 1000000
                               + time_sort_4.tv_usec - time_sort_3.tv_usec)/1e6);

        /* Copy buffers and communicate them to the neighbour ranks */
        create_local_buffers(&mass);

        /* Timer */
        struct timeval time_sort_5;
        gettimeofday(&time_sort_5, NULL);
        message(rank, "Communicating buffers took %.5f s\n",
                               ((time_sort_5.tv_sec - time_sort_4.tv_sec) * 1000000
                               + time_sort_5.tv_usec - time_sort_4.tv_usec)/1e6);

        message(rank, "Computing particle kicks.\n");

        /* Integrate the particles */
        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &particles[i];

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            if (MPI_Rank_Count == 1) {
                accelCIC_single(&mass, N, boxlen, p->x, acc);
            } else {
                accelCIC(&mass, N, boxlen, p->x, acc);
            }

            p->a[0] = acc[0];
            p->a[1] = acc[1];
            p->a[2] = acc[2];

            /* Execute second half-kick */
            p->v[0] += acc[0] * kick_dtau2;
            p->v[1] += acc[1] * kick_dtau2;
            p->v[2] += acc[2] * kick_dtau2;

            /* COLA half-kick */
            if (pars.WithCOLA) {
                p->v[0] += d_vf_next * kick_dtau2 * p->dx[0] / D_start + d_vf_2_next * kick_dtau2 * p->dx2[0] / (D_start * D_start) * factor_vel_2lpt;
                p->v[1] += d_vf_next * kick_dtau2 * p->dx[1] / D_start + d_vf_2_next * kick_dtau2 * p->dx2[1] / (D_start * D_start) * factor_vel_2lpt;
                p->v[2] += d_vf_next * kick_dtau2 * p->dx[2] / D_start + d_vf_2_next * kick_dtau2 * p->dx2[2] / (D_start * D_start) * factor_vel_2lpt;
            }
        }

        /* Step forward */
        a = a_next;

        message(rank, "\n");
    }

    /* Initiate mass deposition */
    if (MPI_Rank_Count == 1) {
        mass_deposition_single(&mass, particles, local_partnum);
    } else {
        mass_deposition(&mass, particles, local_partnum);
    }

    /* Export the GRF */
    writeFieldFile_dg(&mass, "mass.hdf5");

    /* Export a snapshot */
    exportSnapshot(&pars, &us, particles, "snap.hdf5", N, local_partnum);

    /* Free the particle lattice */
    free(particles);

    /* Free the mass grid */
    free_local_grid(&mass);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanParams(&pars);
    cleanCosmology(&cosmo);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);

    /* Clean up the cosmological tables */
    free_cosmology_tables(&ctabs);

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    free_strooklat_spline(&spline_k);
    free_strooklat_spline(&spline_bg_a);

    /* Timer */
    gettimeofday(&time_stop, NULL);
    long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
                           + time_stop.tv_usec - time_start.tv_usec;
    message(rank, "\nTime elapsed: %.5f s\n", microsec/1e6);

}
