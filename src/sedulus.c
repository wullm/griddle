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
            message(rank, "Running with %d MPI ranks.\n", MPI_Rank_Count);
            message(rank, "\n");
        }
    }

    /* Timer */
    struct timepair overall_timer;
    timer_start(rank, &overall_timer);

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
    long int N = pars.PartGridSize;
    double boxlen = pars.BoxLength;
    struct distributed_grid lpt_potential;

    /* Check what portions of 3D grids get stored locally */
    long int X0, NX;
    fftw_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &NX, &X0);

    /* Do the same for the neutrino particles */
    long int N_nu = pars.NeutrinosPerDim;
    long int X0_nu, NX_nu;
    long int local_neutrino_num = 0;
    if (N_nu > 0) {
        fftw_mpi_local_size_3d(N_nu, N_nu, N_nu/2+1, MPI_COMM_WORLD, &NX_nu, &X0_nu);
        local_neutrino_num = NX_nu * N_nu * N_nu;
    }

    /* Each MPI rank stores a portion of the full 3D mesh, as well as copies
     * of the edges of its left & right neighbours, necessary for interpolation
     * and mass deposition. These are called buffers. */
    const int buffer_width = DEFAULT_BUFFER_WIDTH;

    /* The 2LPT factor */
    const double D_start = strooklat_interp(&spline_z, ptdat.D_growth, z_start);
    const double factor_2lpt = 3. / 7.;
    const double factor_vel_2lpt = factor_2lpt * 2.0;

    /* Allocate memory for a particle lattice */
    long long foreign_buffer = pars.ForeignBufferSize; //extra memory for exchanging particles
    long long local_partnum = NX * N * N;
    long long local_firstpart = X0 * N * N;
    long long max_partnum = local_partnum + local_neutrino_num + foreign_buffer;
    struct particle *particles = malloc(max_partnum * sizeof(struct particle));

    if (!pars.GenerateICs) {
        message(rank, "Reading initial conditions from '%s'.\n", pars.InitialConditionsFile);

        if (pars.WithCOLA) {
            printf("COLA is currently only supported with internally generated ICs.\n");
            exit(1);
        }

        /* Timer */
        struct timepair ics_timer;
        timer_start(rank, &ics_timer);

        readSnapshot(&pars, &us, particles, pars.InitialConditionsFile, a_begin, local_partnum, local_firstpart, max_partnum);

        /* Timer */
        timer_stop(rank, &ics_timer, "Reading initial conditions took ");
    } else {
        message(rank, "Generating initial conditions with 2LPT.\n");

        /* Timer */
        struct timepair ics_timer;
        timer_start(rank, &ics_timer);

        /* Allocate distributed memory arrays (one complex & one real) */
        alloc_local_grid(&lpt_potential, N, boxlen, MPI_COMM_WORLD);

        /* Generate LPT potential grid */
        generate_potential_grid(&lpt_potential, &seed, pars.FixedModes,
                                pars.InvertedModes, &ptdat, &cosmo, z_start);

        /* Timer */
        timer_stop(rank, &ics_timer, "Generating the Zel'dovich potential took ");

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

        /* Timer */
        timer_stop(rank, &ics_timer, "Generating the 2LPT potential took ");

        /* Create buffers for the LPT potentials */
        alloc_local_buffers(&lpt_potential, buffer_width);
        alloc_local_buffers(&lpt_potential_2, buffer_width);
        create_local_buffers(&lpt_potential);
        create_local_buffers(&lpt_potential_2);

        /* Generate a particle lattice */
        generate_particle_lattice(&lpt_potential, &lpt_potential_2, &ptdat, &ptpars,
                                  particles, &cosmo, &us, &pcs, X0, NX, z_start);

        /* We are done with the LPT potentials */
        free_local_grid(&lpt_potential);
        free_local_grid(&lpt_potential_2);

        /* Timer */
        timer_stop(rank, &ics_timer, "Generating the particle lattice took ");

        /* Generate neutrino particles */
        if (N_nu > 0) {
            generate_neutrinos(particles, &cosmo, &ctabs, &us, &pcs, N_nu,
                               local_partnum, local_neutrino_num, boxlen,
                               X0, NX, N, z_start, &seed);

            local_partnum += local_neutrino_num;

            /* Timer */
            timer_stop(rank, &ics_timer, "Generating neutrinos took ");
        }
    }

    /* Set velocities to zero when running with COLA */
    if (pars.WithCOLA) {
        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &particles[i];

            if (p->type == 1) {
                p->v[0] = 0.;
                p->v[1] = 0.;
                p->v[2] = 0.;
            }
        }
    }

    /* Make sure that particle coordinates are wrapped */
    for (long long i = 0; i < local_partnum; i++) {
        struct particle *p = &particles[i];
        p->x[0] = fwrap(p->x[0], boxlen);
        p->x[1] = fwrap(p->x[1], boxlen);
        p->x[2] = fwrap(p->x[2], boxlen);
    }

    /* The gravity mesh can be a different size than the particle lattice */
    long int M = pars.MeshGridSize;

    /* Allocate distributed memory arrays for the gravity mesh */
    struct distributed_grid mass;
    alloc_local_grid_with_buffers(&mass, M, boxlen, buffer_width, MPI_COMM_WORLD);
    mass.momentum_space = 0;

    /* Timer */
    struct timepair exchange_timer;
    timer_start(rank, &exchange_timer);

    exchange_particles(particles, boxlen, M, &local_partnum, max_partnum, /* iteration = */ 0, 0, 0);

    /* Timer */
    timer_stop(rank, &exchange_timer, "Exchanging particles took ");
    message(rank, "\n");


    /* Create buffers for the mass grid */
    // alloc_local_buffers(&mass, buffer_width);

    /* First mass deposition */
    if (MPI_Rank_Count == 1) {
        mass_deposition_single(&mass, particles, local_partnum);
    } else {
        mass_deposition(&mass, particles, local_partnum);
    }

    /* Merge the buffers with the main grid */
    add_local_buffers(&mass);

    /* Export the GRF */
    // writeFieldFile_dg(&mass, "ini_mass.hdf5");

    /* Compute the gravitational potential */
    compute_potential(&mass, &pcs);

    /* Copy buffers and communicate them to the neighbour ranks */
    create_local_buffers(&mass);

    /* Determine the snapshot output times */
    double *output_list;
    int num_outputs;
    int last_output = 0;
    parseArrayString(pars.SnapshotTimesString, &output_list, &num_outputs);

    if (num_outputs < 1) {
        printf("No output times specified!\n");
        exit(1);
    } else if (num_outputs > 1) {
        /* Check that the output times are in ascending order */
        for (int i = 1; i < num_outputs; i++) {
            if (output_list[i] <= output_list[i - 1]) {
                printf("Output times should be in strictly ascending order.\n");
                exit(1);
            }
        }
    }

    /* Check that the first output is after the beginning and before the end */
    if (output_list[0] < a_begin) {
        printf("The first output should be after the start of the simulation.\n");
        exit(1);
    } else if (output_list[num_outputs - 1] > a_end) {
        printf("The last output should be before the end of the simulation.\n");
        exit(1);
    }

    /* Check if there should be an output at the start */
    if (output_list[0] == a_begin) {
        exportSnapshot(&pars, &us, particles, 0, a_begin, N, local_partnum);
        last_output++;
    }

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

        /* Conversion factor for neutrino momenta */
        const double fac = pcs.ElectronVolt / (pcs.SpeedOfLight * cosmo.T_nu_0 * pcs.kBoltzmann);

        /* Skip the particle integration during the first step */
        if (ITER == 0)
            continue;

        message(rank, "Step %d at z = %g\n", ITER, z);
        message(rank, "Computing particle kicks and drifts.\n");

        /* Timer */
        struct timepair run_timer;
        timer_start(rank, &run_timer);

        /* Integrate the particles */
        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &particles[i];

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            if (ITER == 1) {
                if (MPI_Rank_Count == 1) {
                    accelCIC_single(&mass, p->x, acc);
                } else {
                    accelCIC(&mass, p->x, acc);
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
            if (pars.WithCOLA && p->type == 1) {
                p->v[0] += d_vf * kick_dtau1 * p->dx[0] / D_start + d_vf_2 * kick_dtau1 * p->dx2[0] / (D_start * D_start) * factor_vel_2lpt;
                p->v[1] += d_vf * kick_dtau1 * p->dx[1] / D_start + d_vf_2 * kick_dtau1 * p->dx2[1] / (D_start * D_start) * factor_vel_2lpt;
                p->v[2] += d_vf * kick_dtau1 * p->dx[2] / D_start + d_vf_2 * kick_dtau1 * p->dx2[2] / (D_start * D_start) * factor_vel_2lpt;
            }

            /* Relativistic drift correction */
            double rel_drift = 1.0;

            /* Neutrino particle operations */
            if (p->type == 6) {
                /* Delta-f weighting for variance reduction (2010.07321) */
                double m_eV = cosmo.M_nu[(int)p->id % cosmo.N_nu];
                double v2 = p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2];
                double q = sqrt(v2) * fac * m_eV;
                double qi = neutrino_seed_to_fermi_dirac(p->id);
                double f = fermi_dirac_density(q);
                double fi = fermi_dirac_density(qi);

                p->w = 1.0 - f / fi;

                /* Relativistic equations of motion (2207.14256) */
                double ac = a * pcs.SpeedOfLight;
                double ac2 = ac * ac;

                rel_drift = ac / sqrt(ac2 + v2);
            }

            /* Execute drift (only one drift, so use dtau = dtau1 + dtau2) */
            p->x[0] += p->v[0] * drift_dtau * rel_drift;
            p->x[1] += p->v[1] * drift_dtau * rel_drift;
            p->x[2] += p->v[2] * drift_dtau * rel_drift;

            /* COLA drift */
            if (pars.WithCOLA && p->type == 1) {
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
        timer_stop(rank, &run_timer, "Evolving particles took ");

        if (MPI_Rank_Count > 1) {

            message(rank, "Starting particle exchange.\n");

            exchange_particles(particles, boxlen, M, &local_partnum, max_partnum, /* iteration = */ 0, 0, 0);

            timer_stop(rank, &run_timer, "Exchanging particles took ");
        }

        message(rank, "Computing the gravitational potential.\n");

        /* Initiate mass deposition */
        if (MPI_Rank_Count == 1) {
            mass_deposition_single(&mass, particles, local_partnum);
        } else {
            mass_deposition(&mass, particles, local_partnum);
        }

        /* Timer */
        timer_stop(rank, &run_timer, "Computing mass density took ");

        /* Merge the buffers with the main grid */
        add_local_buffers(&mass);

        /* Timer */
        timer_stop(rank, &run_timer, "Communicating buffers took ");

        /* Re-compute the gravitational potential */
        compute_potential(&mass, &pcs);

        /* Timer */
        timer_stop(rank, &run_timer, "Computing the potential took ");

        /* Copy buffers and communicate them to the neighbour ranks */
        create_local_buffers(&mass);

        /* Timer */
        timer_stop(rank, &run_timer, "Communicating buffers took ");

        message(rank, "Computing particle kicks.\n");

        /* Integrate the particles */
        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &particles[i];

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            if (MPI_Rank_Count == 1) {
                accelCIC_single(&mass, p->x, acc);
            } else {
                accelCIC(&mass, p->x, acc);
            }

            p->a[0] = acc[0];
            p->a[1] = acc[1];
            p->a[2] = acc[2];

            /* Execute second half-kick */
            p->v[0] += acc[0] * kick_dtau2;
            p->v[1] += acc[1] * kick_dtau2;
            p->v[2] += acc[2] * kick_dtau2;

            /* COLA half-kick */
            if (pars.WithCOLA && p->type == 1) {
                p->v[0] += d_vf_next * kick_dtau2 * p->dx[0] / D_start + d_vf_2_next * kick_dtau2 * p->dx2[0] / (D_start * D_start) * factor_vel_2lpt;
                p->v[1] += d_vf_next * kick_dtau2 * p->dx[1] / D_start + d_vf_2_next * kick_dtau2 * p->dx2[1] / (D_start * D_start) * factor_vel_2lpt;
                p->v[2] += d_vf_next * kick_dtau2 * p->dx[2] / D_start + d_vf_2_next * kick_dtau2 * p->dx2[2] / (D_start * D_start) * factor_vel_2lpt;
            }
        }


        timer_stop(rank, &run_timer, "Computing particle kicks took ");

        /* Should there be a snapshot output? */
        while (output_list[last_output] > a && output_list[last_output] <= a_next) {

            message(rank, "Exporting a snapshot at a = %g.\n", output_list[last_output]);
            exportSnapshot(&pars, &us, particles, last_output, output_list[last_output], N, local_partnum);
            last_output++;

            timer_stop(rank, &run_timer, "Exporting a snapshot took took ");
        }

        /* Step forward */
        a = a_next;

        message(rank, "\n");
    }

    // /* Initiate mass deposition */
    // if (MPI_Rank_Count == 1) {
    //     mass_deposition_single(&mass, particles, local_partnum);
    // } else {
    //     mass_deposition(&mass, particles, local_partnum);
    // }

    /* Export the GRF */
    // writeFieldFile_dg(&mass, "final_mass.hdf5");

    /* Free the list with snapshot output times */
    free(output_list);

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
    message(rank, "\n");
    timer_stop(rank, &overall_timer, "Time elapsed: ");
}
