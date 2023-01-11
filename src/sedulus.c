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

// #define DEBUG_CHECKS

int main(int argc, char *argv[]) {
    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
#ifdef SINGLE_PRECISION_FFTW
    fftwf_mpi_init();
#else
    fftw_mpi_init();
#endif

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
        message(rank, "# GIT branch: %s\n", GIT_BRANCH);
        message(rank, "# GIT commit: %s\n", GIT_COMMIT);
        message(rank, "# GIT message: %s\n", GIT_MESSAGE);
        message(rank, "# GIT date: %s\n", GIT_DATE);
        message(rank, "# GIT status:\n");
        message(rank, "\n");
        message(rank, "%s\n", GIT_STATUS);
        message(rank, "\n");
        message(rank, "sizeof(struct particle) = %d\n", sizeof(struct particle));
        message(rank, "sizeof(GridFloatType) = %d\n", sizeof(GridFloatType));
        message(rank, "sizeof(GridComplexType) = %d\n", sizeof(GridComplexType));
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
    const double a_target = pars.ScaleFactorTarget;
    const double a_step = 1.0 + pars.ScaleFactorStep;
    const double z_start = 1.0 / a_begin - 1.0;
    const double z_target = 1.0 / a_target - 1.0;

    /* Timer */
    struct timepair setup_timer;
    timer_start(rank, &setup_timer);

    /* Integrate the background cosmology */
    integrate_cosmology_tables(&cosmo, &us, &pcs, &ctabs, 1e-3, fmax(1.0, a_end) * 1.01, 1000);

    /* Timer */
    timer_stop(rank, &setup_timer, "Integrating cosmological tables took ");

    /* Read the perturbation data file */
    readPerturb(&us, &ptdat, pars.TransferFunctionsFile);
    readPerturbParams(&us, &ptpars, pars.TransferFunctionsFile);

    /* Timer */
    timer_stop(rank, &setup_timer, "Reading transfer functions took ");

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat.redshift, ptdat.tau_size};
    struct strooklat spline_k = {ptdat.k, ptdat.k_size};
    init_strooklat_spline(&spline_z, 100);
    init_strooklat_spline(&spline_k, 100);

    /* Additional interpolation spline for the background cosmology scale factors */
    struct strooklat spline_bg_a = {ctabs.avec, ctabs.size};
    init_strooklat_spline(&spline_bg_a, 100);

    /* Perform a Newtonian back-scaling procedure on the transfer functions */
    double f_asymptotic = 0.; // the asymptotic logarithmic growth rate
    if (pars.DoNewtonianBackscaling) {
        backscale_transfers(&ptdat, &cosmo, &ctabs, &us, &pcs, z_start, z_target, &f_asymptotic);
        timer_stop(rank, &setup_timer, "Solving back-scaling fluid equations took ");
    }

    /* Store the MPI rank */
    pars.rank = rank;

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed + rank);

    /* Create or read a Gaussian random field */
    long int N = pars.PartGridSize;
    double boxlen = pars.BoxLength;

    /* Check what portions of 3D grids get stored locally */
    long int X0, NX;
    fft_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &NX, &X0);

    /* Do the same for the neutrino particles */
    long int N_nu = pars.NeutrinosPerDim;
    long int X0_nu, NX_nu;
    long int local_neutrino_num = 0;
    if (N_nu > 0) {
        fft_mpi_local_size_3d(N_nu, N_nu, N_nu/2+1, MPI_COMM_WORLD, &NX_nu, &X0_nu);
        local_neutrino_num = NX_nu * N_nu * N_nu;
    }

#ifdef WITH_PARTICLE_IDS
#ifdef SINGLE_PRECISION_IDS
    if (N * N * N + N_nu * N_nu * N_nu > UINT32_MAX) {
        printf("Number of particles exceeds UINT32_MAX. Please disable SINGLE_PRECISION_IDS.\n");
        exit(1);
    }
#else
    if (N * N * N + N_nu * N_nu * N_nu > UINT64_MAX) {
        printf("Number of particles exceeds UINT64_MAX. Perhaps add long int ids.\n");
        exit(1);
    }
#endif
#endif

    /* Each MPI rank stores a portion of the full 3D mesh, as well as copies
     * of the edges of its left & right neighbours, necessary for interpolation
     * and mass deposition. These are called buffers. */
    const int buffer_width = DEFAULT_BUFFER_WIDTH;

    /* Allocate memory for a particle lattice */
    long long foreign_buffer = pars.ForeignBufferSize; //extra memory for exchanging particles
    long long local_cdm_num = NX * N * N;
    long long local_firstpart = X0 * N * N;
    long long max_partnum = local_cdm_num + local_neutrino_num + foreign_buffer;
    struct particle *particles = malloc(max_partnum * sizeof(struct particle));

    /* The current number of particles */
    long long local_partnum = 0;

    if (!pars.GenerateICs) {
        message(rank, "Reading initial conditions from '%s'.\n", pars.InitialConditionsFile);

        /* Timer */
        struct timepair ics_timer;
        timer_start(rank, &ics_timer);

        local_partnum = local_cdm_num;
        readSnapshot(&pars, &us, particles, pars.InitialConditionsFile, a_begin, local_partnum, local_firstpart, max_partnum);

        /* Timer */
        timer_stop(rank, &ics_timer, "Reading initial conditions took ");
    } else {

        /* Timer */
        struct timepair ics_timer;
        timer_start(rank, &ics_timer);

        /* Generate neutrino particles first */
        if (N_nu > 0) {
            message(rank, "Generating neutrino initial conditions.\n");

            generate_neutrinos(particles, &cosmo, &ctabs, &us, &pcs, N_nu,
                               local_partnum, local_cdm_num, local_neutrino_num,
                               boxlen, X0_nu, NX_nu, z_start, &seed);

            local_partnum += local_neutrino_num;

            /* Timer */
            timer_stop(rank, &ics_timer, "Generating neutrinos took ");

            if (pars.NeutrinoPreIntegration) {
                /* Allocate distributed memory arrays for a potential grid */
                struct distributed_grid nu_potential;
                alloc_local_grid_with_buffers(&nu_potential, N_nu, boxlen, buffer_width, MPI_COMM_WORLD);

                pre_integrate_neutrinos(&nu_potential, &ptdat, &pars, pars.FixedModes,
                                        pars.InvertedModes, particles, &cosmo, &ctabs,
                                        &us, &pcs, N_nu, local_partnum, max_partnum,
                                        local_neutrino_num, boxlen, z_start, pars.Seed);

                free_local_grid(&nu_potential);

                /* Timer */
                timer_stop(rank, &ics_timer, "Integrating neutrinos took ");
            }
        }

        message(rank, "Generating initial conditions with 2LPT.\n");

        /* Allocate distributed memory arrays for the LPT potential */
        struct distributed_grid lpt_potential;
        alloc_local_grid_with_buffers(&lpt_potential, N, boxlen, buffer_width, MPI_COMM_WORLD);

        /* Generate LPT potential grid */
        generate_potential_grid(&lpt_potential, pars.Seed, pars.FixedModes,
                                pars.InvertedModes, density_potential_type,
                                &ptdat, &cosmo, z_start);

        /* Timer */
        timer_stop(rank, &ics_timer, "Generating the Zel'dovich potential took ");

        /* Allocate additional arrays */
        struct distributed_grid temp2;
        struct distributed_grid temp1;
        struct distributed_grid lpt_potential_2;
        alloc_local_grid(&temp1, N, boxlen, MPI_COMM_WORLD);
        alloc_local_grid(&temp2, N, boxlen, MPI_COMM_WORLD);
        alloc_local_grid_with_buffers(&lpt_potential_2, N, boxlen, buffer_width, MPI_COMM_WORLD);

        /* Generate the 2LPT potential grid */
        generate_2lpt_grid(&lpt_potential, &temp1, &temp2, &lpt_potential_2,
                           &ptdat, &cosmo, z_start);

        /* Free working memory used in the 2LPT calculation */
        free_local_grid(&temp1);
        free_local_grid(&temp2);

        /* Timer */
        timer_stop(rank, &ics_timer, "Generating the 2LPT potential took ");

        /* Create buffers for the LPT potentials */
        create_local_buffers(&lpt_potential);
        create_local_buffers(&lpt_potential_2);

        /* Allocate another array for the linear theory velocity field */
        struct distributed_grid velocity_potential;
        alloc_local_grid_with_buffers(&velocity_potential, N, boxlen, buffer_width, MPI_COMM_WORLD);

        /* Generate velocity potential */
        generate_potential_grid(&velocity_potential, pars.Seed, pars.FixedModes,
                                pars.InvertedModes, velocity_potential_type,
                                &ptdat, &cosmo, z_start);

        /* Create buffer for the velocity potential */
        create_local_buffers(&velocity_potential);

        /* Timer */
        timer_stop(rank, &ics_timer, "Generating the velocity potential took ");

        /* Generate a particle lattice for the dark matter */
        generate_particle_lattice(&lpt_potential, &lpt_potential_2,
                                  &velocity_potential, &ptdat, particles,
                                  &cosmo, &us, &pcs, local_partnum, X0, NX,
                                  z_start, f_asymptotic);

        local_partnum += local_cdm_num;

        /* We are done with the LPT and velocity potentials */
        free_local_grid(&lpt_potential);
        free_local_grid(&lpt_potential_2);
        free_local_grid(&velocity_potential);

        /* Timer */
        timer_stop(rank, &ics_timer, "Generating the particle lattice took ");
    }

    /* The gravity mesh can be a different size than the particle lattice */
    long int M = pars.MeshGridSize;

    /* Allocate distributed memory arrays for the gravity mesh */
    struct distributed_grid mass;
    alloc_local_grid_with_buffers(&mass, M, boxlen, buffer_width, MPI_COMM_WORLD);
    mass.momentum_space = 0;

    if (MPI_Rank_Count > 1) {
        /* Timer */
        struct timepair exchange_timer;
        timer_start(rank, &exchange_timer);

        exchange_particles(particles, boxlen, M, &local_partnum, max_partnum, /* iteration = */ 0, 0, 0, 0, 0);

        /* Timer */
        timer_stop(rank, &exchange_timer, "Exchanging particles took ");
    }
    message(rank, "\n");

    /* Timer */
    struct timepair fft_plan_timer;
    timer_start(rank, &fft_plan_timer);

    /* Prepare FFT plans for the gravity mesh */
    message(rank, "Carefully planning FFTs. This may take awhile.\n");
    FourierPlanType r2c_mpi, c2r_mpi;
    fft_prepare_mpi_plans(&r2c_mpi, &c2r_mpi, &mass);

    /* Timer */
    timer_stop(rank, &fft_plan_timer, "Planning FFTs took ");
    message(rank, "\n");

#if defined(WITH_PARTTYPE) && defined(WITH_PARTICLE_IDS)
    /* Conversion factor for neutrino momenta */
    const double neutrino_qfac = pcs.ElectronVolt / (pcs.SpeedOfLight * cosmo.T_nu_0 * pcs.kBoltzmann);

    /* Create a file with neutrino weight information for each step */
    char fname_nu_info[50] = "neutrino_weights_info.txt";
    double neutrino_weights_sq_mean = 0.;
    double nu_shot_factor = 1.0;

    if (N_nu > 0) {
        /* Create the file */
        if (rank == 0) {
            FILE *f = fopen(fname_nu_info, "w");
            fprintf(f, "# Step a z <w> <w^2> <q> number\n");
            fclose(f);
        }

        /* Accumulate the neutrino weights */
        long int neutrino_count_local = 0;
        double weights_sq_sum_local = 0.;

        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &particles[i];
            if (p->type == 6) {
                neutrino_count_local++;
                weights_sq_sum_local += p->w * p->w;
            }
        }

        /* Collect neutrino weight information from all ranks */
        long int neutrino_count_global;
        double weights_sq_sum_global;
        MPI_Reduce(&neutrino_count_local, &neutrino_count_global, 1,
                   MPI_LONG, MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
        MPI_Reduce(&weights_sq_sum_local, &weights_sq_sum_global, 1,
                   MPI_DOUBLE, MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);

        /* Communicate the mean squared weight */
        neutrino_weights_sq_mean = weights_sq_sum_global / neutrino_count_global;
        MPI_Bcast(&neutrino_weights_sq_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        nu_shot_factor = neutrino_weights_sq_mean;
    }
#else
    const double nu_shot_factor = 1.0;
#endif

    /* Determine the output times for various output types */
    double *output_list_snap, *output_list_power, *output_list_halos;
    int num_outputs_snap, num_outputs_power, num_outputs_halos;
    parseOutputList(pars.SnapshotTimesString, &output_list_snap,
                    &num_outputs_snap, a_begin, a_end);
    parseOutputList(pars.PowerSpectrumTimesString, &output_list_power,
                    &num_outputs_power, a_begin, a_end);
    parseOutputList(pars.HaloFindingTimesString, &output_list_halos,
                    &num_outputs_halos, a_begin, a_end);

    message(rank, "Exporting snapshots %d times.\n", num_outputs_snap);
    message(rank, "Exporting power spectra %d times.\n", num_outputs_power);
    message(rank, "Exporting halo catalogues %d times.\n", num_outputs_halos);
    message(rank, "\n");

    /* Parse the power spectrum types */
    enum grid_type *powspec_types = NULL;
    int powspec_types_num;
    parseGridTypeList(pars.PowerSpectrumTypes, &powspec_types, &powspec_types_num);

    if (powspec_types_num > 0) {
        message(rank, "Power spectrum types: ");
        for (int i = 0; i < powspec_types_num; i++) {
            message(rank, "%s%s", grid_type_names[powspec_types[i]], (i < powspec_types_num - 1) ? ", " : "\n");
        }
        message(rank, "\n");
    }

    /* Check if there should be an output at the start */
    if (num_outputs_power > 0 && output_list_power[0] == a_begin) {
        /* Timer */
        struct timepair powspec_timer;
        timer_start(rank, &powspec_timer);

        for (int i = 0; i < powspec_types_num; i++) {
            message(rank, "Starting power spectrum calculation (%s).\n",
                          grid_type_names[powspec_types[i]]);

            /* Initiate mass deposition */
            mass_deposition(&mass, particles, local_partnum, powspec_types[i]);
            timer_stop(rank, &powspec_timer, "Computing mass density took ");

            /* Merge the buffers with the main grid */
            add_local_buffers(&mass);
            timer_stop(rank, &powspec_timer, "Communicating buffers took ");

            /* Compute position-dependent power spectra */
            if (powspec_types[i] == all_mass) {
                analysis_posdep(&mass, /* output_num = */ 0, a_begin, &us, &pcs,
                                &cosmo, &pars);
                timer_stop(rank, &powspec_timer, "Position-dependent power spectra took ");
            }

            /* Compute the level of shot noise */
            const double shot_noise = calc_shot_noise(boxlen, N, N_nu, nu_shot_factor, powspec_types[i]);

            /* Compute global power spectra */
            analysis_powspec(&mass, /* output_num = */ 0, a_begin, r2c_mpi, &us,
                             &pcs, &cosmo, &pars, powspec_types[i], shot_noise);
            timer_stop(rank, &powspec_timer, "Global power spectra took ");

            message(rank, "\n");
        }
    }

    if (num_outputs_halos > 0 && output_list_halos[0] == a_begin) {
        /* Timer */
        struct timepair fof_timer;
        timer_start(rank, &fof_timer);

        message(rank, "Starting friends-of-friends halo finding.\n");

        /* Free the main grid before engaging the halo finder */
        free_local_grid(&mass);

        /* Run the halo finder */
        const double linking_length = pars.LinkingLength * boxlen / N;
        analysis_fof(particles, boxlen, N, M, local_partnum, max_partnum, linking_length, pars.MinHaloParticleNum, /* output_num = */ 0, a_begin, &us, &pcs, &cosmo, &pars, &ctabs,  /* kick_dtau = */ 0., /* drift_dtau = */ 0.);

        /* Timer */
        MPI_Barrier(MPI_COMM_WORLD);
        timer_stop(rank, &fof_timer, "Finding halos took ");
        message(rank, "\n");

        /* Re-allocate the main grid after finishing with the halo finder */
        alloc_local_grid_with_buffers(&mass, M, boxlen, buffer_width, MPI_COMM_WORLD);
        mass.momentum_space = 0;

        /* Re-create the FFT plans */
        fft_prepare_mpi_plans(&r2c_mpi, &c2r_mpi, &mass);
    }

    if (num_outputs_snap > 0 && output_list_snap[0] == a_begin) {
        /* Timer */
        struct timepair snapshot_timer;
        timer_start(rank, &snapshot_timer);

        message(rank, "Exporting a snapshot at a = %g.\n", output_list_snap[0]);
        exportSnapshot(&pars, &us, &pcs, particles, 0, a_begin, N, local_partnum, /* kick_dtau = */ 0., /* drift_dtau = */ 0.);

        timer_stop(rank, &snapshot_timer, "Exporting a snapshot took ");
        message(rank, "\n");
    }

#ifndef WITH_MASSES
    /* If particles do not have individual masses, then the mass density grid
     * is instead a number density grid. We then multiply the accelerations by
     * the appropriate factor in the main loop. */
    const double h = cosmo.h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us.UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs.GravityG);
    const double Omega_m = cosmo.Omega_cdm + cosmo.Omega_b;
    const double part_mass = rho_crit * Omega_m * pow(boxlen / N, 3);
#endif

    /* Position factors */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;

    /* Position factors on [0, M] where M is the PM size */
    const double grid_to_int_fac = pow(2.0, POSITION_BITS) / M;
    const double int_to_grid_fac = 1.0 / grid_to_int_fac;
    const double cell_fac = M / boxlen;

    /* Pointer to the real-space potential grid */
    const GridFloatType *mass_box = mass.buffered_box;
    const long int Mz = mass.Nz;
    const long int MX0 = mass.X0;

    /* Start at the beginning */
    double a = a_begin;

    /* Prepare integration */
    const int MAX_ITER = (log(a_end) - log(a_begin))/log(a_step) + 1;
    const double a_factor = exp(log(a_end / a_begin) / MAX_ITER);

    /* The main loop */
    for (int ITER = 0; ITER < MAX_ITER; ITER++) {

        /* Determine the previous and next scale factor */
        double a_prev, a_next;
        if (ITER == 0) {
            a_prev = a;
            a_next = a * a_factor;
        } else if (ITER < MAX_ITER - 1) {
            a_prev = a / a_factor;
            a_next = a * a_factor;
        } else {
            a_prev = a / a_factor;
            a_next = a_end;
        }

        /* Compute the current redshift and log conformal time */
        double z = 1./a - 1.;

        /* Determine the adjacent half-step scale factor */
        double a_half_prev = sqrt(a_prev * a);
        double a_half_next = sqrt(a_next * a);

        /* Obtain the kick and drift time steps */
        double kick_dtau  = strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_half_next) -
                            strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_half_prev);
        double drift_dtau = strooklat_interp(&spline_bg_a, ctabs.drift_factors, a_next) -
                            strooklat_interp(&spline_bg_a, ctabs.drift_factors, a);

#if defined(WITH_PARTTYPE) && defined(WITH_PARTICLE_IDS)
        /* Accumulate the neutrino weights */
        long int neutrino_count_local = 0;
        double weights_sum_local = 0.;
        double weights_sq_sum_local = 0.;
        double neutrino_q_sum_local = 0.;
#endif

        message(rank, "Step %d at z = %g\n", ITER, z);

        /* Timer */
        struct timepair run_timer;
        timer_start(rank, &run_timer);

        /* Initiate mass deposition */
        mass_deposition(&mass, particles, local_partnum, all_mass);
        timer_stop(rank, &run_timer, "Computing mass density took ");

        /* Merge the buffers with the main grid */
        add_local_buffers(&mass);
        timer_stop(rank, &run_timer, "Communicating buffers took ");

        /* Re-compute the gravitational potential */
        compute_potential(&mass, &pcs, r2c_mpi, c2r_mpi);
        timer_stop(rank, &run_timer, "Computing the potential in total took ");

        /* Copy buffers and communicate them to the neighbour ranks */
        create_local_buffers(&mass);
        timer_stop(rank, &run_timer, "Communicating buffers took ");

        message(rank, "Computing particle kicks and drifts.\n");

        /* Integrate the particles */
        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &particles[i];

            /* Convert integer positions to floating points on [0, M] */
            double x[3] = {p->x[0] * int_to_grid_fac,
                           p->x[1] * int_to_grid_fac,
                           p->x[2] * int_to_grid_fac};

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            if (pars.DerivativeOrder == 1) {
                accelCIC_1st(mass_box, x, acc, M, MX0, buffer_width, Mz, cell_fac); /* first order */
            } else if (pars.DerivativeOrder == 2) {
                accelCIC_2nd(mass_box, x, acc, M, MX0, buffer_width, Mz, cell_fac); /* second order */
            } else if (pars.DerivativeOrder == 4) {
                accelCIC_4th(mass_box, x, acc, M, MX0, buffer_width, Mz, cell_fac); /* fourth order */
            } else {
                printf("Differentiation scheme with order %d not implemented.\n", pars.DerivativeOrder);
            }

#ifndef WITH_MASSES
            acc[0] *= part_mass;
            acc[1] *= part_mass;
            acc[2] *= part_mass;
#endif

#ifdef WITH_ACCELERATIONS
            p->a[0] = acc[0];
            p->a[1] = acc[1];
            p->a[2] = acc[2];
#endif

            /* Execute kick */
            FloatVelType v[3] = {p->v[0] + acc[0] * kick_dtau,
                                 p->v[1] + acc[1] * kick_dtau,
                                 p->v[2] + acc[2] * kick_dtau};

            /* Relativistic drift correction */
            const double rel_drift = relativistic_drift(v, p, &pcs, a);

            /* Execute drift */
            p->x[0] += v[0] * drift_dtau * rel_drift * pos_to_int_fac;
            p->x[1] += v[1] * drift_dtau * rel_drift * pos_to_int_fac;
            p->x[2] += v[2] * drift_dtau * rel_drift * pos_to_int_fac;

            /* Update velocities */
            p->v[0] = v[0];
            p->v[1] = v[1];
            p->v[2] = v[2];

            /* Delta-f weighting for neutrino variance reduction (2010.07321) */
#if defined(WITH_PARTTYPE) && defined(WITH_PARTICLE_IDS)
            if (p->type == 6) {
                double m_eV = cosmo.M_nu[(int)p->id % cosmo.N_nu];
                double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                double q = sqrt(v2) * neutrino_qfac * m_eV;
                double qi = neutrino_seed_to_fermi_dirac(p->id);
                double f = fermi_dirac_density(q);
                double fi = fermi_dirac_density(qi);

                p->w = 1.0 - f / fi;

                neutrino_count_local++;
                weights_sum_local += p->w;
                weights_sq_sum_local += p->w * p->w;
                neutrino_q_sum_local += q;
            }
#endif

            // /* Convert positions to integers (wrapping automatic by overflow) */
            // p->x[0] = x[0] * grid_to_int_fac;
            // p->x[1] = x[1] * grid_to_int_fac;
            // p->x[2] = x[2] * grid_to_int_fac;
        }

        /* Timer */
        timer_stop(rank, &run_timer, "Evolving particles took ");

#if defined(WITH_PARTTYPE) && defined(WITH_PARTICLE_IDS)
        if (N_nu > 0) {
            /* Collect neutrino weight information from all ranks */
            long int neutrino_count_global;
            double local_quantities[3] = {weights_sum_local,
                                          weights_sq_sum_local,
                                          neutrino_q_sum_local};
            double global_quantities[3];
            MPI_Reduce(&neutrino_count_local, &neutrino_count_global, 1,
                       MPI_LONG, MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
            MPI_Reduce(local_quantities, global_quantities, 3, MPI_DOUBLE,
                       MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);

            /* Print it to a file */
            if (rank == 0) {
                double weights_mean = global_quantities[0] / neutrino_count_global;
                double weights_sq_mean = global_quantities[1] / neutrino_count_global;
                double neutrino_q_mean = global_quantities[2] / neutrino_count_global;

                neutrino_weights_sq_mean = weights_sq_mean;

                FILE *f = fopen(fname_nu_info, "a");
                fprintf(f, "%d %g %g %g %g %g %ld\n", ITER, a, z, weights_mean, weights_sq_mean, neutrino_q_mean, neutrino_count_global);
                fclose(f);
            }

            /* Communicate the mean squared weight */
            MPI_Bcast(&neutrino_weights_sq_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            nu_shot_factor = neutrino_weights_sq_mean;

            timer_stop(rank, &run_timer, "Writing neutrino information took ");
        }
#endif

        /* Should there be a power spectrum output? */
        for (int j = 0; j < num_outputs_power; j++) {
            if (output_list_power[j] > a && output_list_power[j] <= a_next) {

                /* Drift and kick particles to the right time */
                // double power_kick_dtau  = strooklat_interp(&spline_bg_a, ctabs.kick_factors, output_list_power[j]) -
                //                           strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_half_next);
                double power_drift_dtau  = strooklat_interp(&spline_bg_a, ctabs.kick_factors, output_list_power[j]) -
                                           strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_next);

                message(rank, "\n");
                message(rank, "Starting power spectrum calculation at a = %g.\n",
                              output_list_power[j]);

#ifdef DEBUG_CHECKS
                /* Compute position checksum */
                IntPosType checksum_global_before = position_checksum(particles, local_partnum);
                timer_stop(rank, &run_timer, "Computing checksum took ");
#endif

                /* Reversibly drift the particles to the correct time */
                drift_particles(particles, local_partnum, a, power_drift_dtau, pos_to_int_fac, &pcs);
                timer_stop(rank, &run_timer, "Drifting particles to output time took ");

                /* Reversibly kick neutrino particle weights */
                if (N_nu > 0) {
                    kick_weights_only(particles, local_partnum, a, power_drift_dtau, neutrino_qfac, &cosmo, &pcs);
                    timer_stop(rank, &run_timer, "Kicking particle weights to output time took ");
                }
                message(rank, "\n");

                for (int i = 0; i < powspec_types_num; i++) {
                    message(rank, "Working on power spectrum type '%s'.\n",
                                  output_list_power[j], grid_type_names[powspec_types[i]]);

                    /* Exchange particles before attempting a mass deposition */
                    exchange_particles(particles, boxlen, M, &local_partnum, max_partnum, /* iteration = */ 0, 0, 0, 0, 0);
                    timer_stop(rank, &run_timer, "Exchanging particles took ");

                    /* Initiate mass deposition */
                    mass_deposition(&mass, particles, local_partnum, powspec_types[i]);
                    timer_stop(rank, &run_timer, "Computing mass density took ");

                    /* Merge the buffers with the main grid */
                    add_local_buffers(&mass);
                    timer_stop(rank, &run_timer, "Communicating buffers took ");

                    /* Compute position-dependent power spectra */
                    if (powspec_types[i] == all_mass) {
                        analysis_posdep(&mass, /* output_num = */ j, output_list_power[j], &us, &pcs, &cosmo, &pars);
                        timer_stop(rank, &run_timer, "Position-dependent power spectra took ");
                    }

                    /* Compute the level of shot noise */
                    const double shot_noise = calc_shot_noise(boxlen, N, N_nu, nu_shot_factor, powspec_types[i]);

                    analysis_powspec(&mass, /* output_num = */ j, output_list_power[j], r2c_mpi, &us, &pcs, &cosmo, &pars, powspec_types[i], shot_noise);
                    timer_stop(rank, &run_timer, "Global power spectra took ");

                    message(rank, "\n");
                }

                /* Reverse the particle drifts */
                drift_particles(particles, local_partnum, a, -power_drift_dtau, pos_to_int_fac, &pcs);
                timer_stop(rank, &run_timer, "Reversing particle drift took ");

                if (N_nu > 0) {
                    kick_weights_only(particles, local_partnum, a, /* kick_dtau = */ 0., neutrino_qfac, &cosmo, &pcs);
                    timer_stop(rank, &run_timer, "Reversing weight kicks took ");
                }

#ifdef DEBUG_CHECKS
                /* Compute position checksum */
                IntPosType checksum_global_after = position_checksum(particles, local_partnum);
                timer_stop(rank, &run_timer, "Computing checksum took ");
                message(rank, "Checksum: %u -> %u\n", checksum_global_before, checksum_global_after);
                if (rank == 0) assert(checksum_global_before == checksum_global_after);
#endif
            }
        }

        /* Should there be a halo finding output? */
        for (int j = 0; j < num_outputs_halos; j++) {
            if (output_list_halos[j] > a && output_list_halos[j] <= a_next) {

                /* Drift and kick particles to the right time */
                double halos_kick_dtau  = strooklat_interp(&spline_bg_a, ctabs.kick_factors, output_list_halos[j]) -
                                          strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_half_next);
                double halos_drift_dtau  = strooklat_interp(&spline_bg_a, ctabs.kick_factors, output_list_halos[j]) -
                                           strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_next);

                message(rank, "\n");
                message(rank, "Starting friends-of-friends halo finding at a = %g.\n", output_list_halos[j]);

#ifdef DEBUG_CHECKS
                /* Compute position checksum */
                IntPosType checksum_global_before = position_checksum(particles, local_partnum);
                timer_stop(rank, &run_timer, "Computing checksum took ");
#endif

                /* Reversibly drift the particles to the correct time */
                drift_particles(particles, local_partnum, a, halos_drift_dtau, pos_to_int_fac, &pcs);
                timer_stop(rank, &run_timer, "Drifting particles to output time took ");

                /* Reversibly kick neutrino particle weights */
                if (N_nu > 0) {
                    kick_weights_only(particles, local_partnum, a, halos_drift_dtau, neutrino_qfac, &cosmo, &pcs);
                    timer_stop(rank, &run_timer, "Kicking particle weights to output time took ");
                }
                message(rank, "\n");

                /* Free the main grid before engaging the halo finder */
                free_local_grid(&mass);

                /* Run the halo finder */
                const double linking_length = pars.LinkingLength * boxlen / N;
                analysis_fof(particles, boxlen, N, M, local_partnum, max_partnum, linking_length, pars.MinHaloParticleNum, /* output_num = */ j, output_list_halos[j], &us, &pcs, &cosmo, &pars, &ctabs, halos_kick_dtau, halos_drift_dtau);

                /* Timer */
                MPI_Barrier(MPI_COMM_WORLD);
                timer_stop(rank, &run_timer, "Finding halos took ");
                message(rank, "\n");

                /* Reverse the particle drifts */
                drift_particles(particles, local_partnum, a, -halos_drift_dtau, pos_to_int_fac, &pcs);
                timer_stop(rank, &run_timer, "Reversing particle drift took ");

                if (N_nu > 0) {
                    kick_weights_only(particles, local_partnum, a, /* kick_dtau = */ 0., neutrino_qfac, &cosmo, &pcs);
                    timer_stop(rank, &run_timer, "Reversing weight kicks took ");
                }

#ifdef DEBUG_CHECKS
                /* Compute position checksum */
                IntPosType checksum_global_after = position_checksum(particles, local_partnum);
                timer_stop(rank, &run_timer, "Computing checksum took ");
                message(rank, "Checksum: %u -> %u\n", checksum_global_before, checksum_global_after);
                if (rank == 0) assert(checksum_global_before == checksum_global_after);
#endif

                /* Re-allocate the main grid after finishing with the halo finder */
                alloc_local_grid_with_buffers(&mass, M, boxlen, buffer_width, MPI_COMM_WORLD);
                mass.momentum_space = 0;

                /* Re-create the FFT plans */
                fft_prepare_mpi_plans(&r2c_mpi, &c2r_mpi, &mass);
                timer_stop(rank, &run_timer, "Recreating Fourier structures took ");
                message(rank, "\n");
            }
        }

        /* Should there be a snapshot output? */
        for (int j = 0; j < num_outputs_snap; j++) {
            if (output_list_snap[j] > a && output_list_snap[j] <= a_next) {

                /* Drift and kick particles to the right time */
                double snap_kick_dtau  = strooklat_interp(&spline_bg_a, ctabs.kick_factors, output_list_snap[j]) -
                                         strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_half_next);
                double snap_drift_dtau  = strooklat_interp(&spline_bg_a, ctabs.kick_factors, output_list_snap[j]) -
                                          strooklat_interp(&spline_bg_a, ctabs.kick_factors, a_next);

                /* Free the main grid before writing a full snapshot */
                free_local_grid(&mass);

                message(rank, "Exporting a snapshot at a = %g.\n", output_list_snap[j]);

#ifdef DEBUG_CHECKS
                /* Compute position checksum */
                IntPosType checksum_global_before = position_checksum(particles, local_partnum);
                timer_stop(rank, &run_timer, "Computing checksum took ");
#endif

                /* Reversibly drift the particles to the correct time */
                drift_particles(particles, local_partnum, a, snap_drift_dtau, pos_to_int_fac, &pcs);
                timer_stop(rank, &run_timer, "Drifting particles to output time took ");

                /* Reversibly kick neutrino particle weights */
                if (N_nu > 0) {
                    kick_weights_only(particles, local_partnum, a, snap_kick_dtau, neutrino_qfac, &cosmo, &pcs);
                    timer_stop(rank, &run_timer, "Kicking particle weights to output time took ");
                }
                message(rank, "\n");

                exportSnapshot(&pars, &us, &pcs, particles, /* output_num = */ j, output_list_snap[j], N, local_partnum, snap_kick_dtau, snap_drift_dtau);
                timer_stop(rank, &run_timer, "Exporting a snapshot took ");

                /* Reverse the particle drifts */
                drift_particles(particles, local_partnum, a, -snap_drift_dtau, pos_to_int_fac, &pcs);
                timer_stop(rank, &run_timer, "Reversing particle drift took ");

                if (N_nu > 0) {
                    kick_weights_only(particles, local_partnum, a, /* kick_dtau = */ 0., neutrino_qfac, &cosmo, &pcs);
                    timer_stop(rank, &run_timer, "Reversing weight kicks took ");
                }

#ifdef DEBUG_CHECKS
                /* Compute position checksum */
                IntPosType checksum_global_after = position_checksum(particles, local_partnum);
                timer_stop(rank, &run_timer, "Computing checksum took ");
                message(rank, "Checksum: %u -> %u\n", checksum_global_before, checksum_global_after);
                if (rank == 0) assert(checksum_global_before == checksum_global_after);
#endif

                /* Re-allocate the main grid after writing the snapshot */
                alloc_local_grid_with_buffers(&mass, M, boxlen, buffer_width, MPI_COMM_WORLD);
                mass.momentum_space = 0;

                /* Re-create the FFT plans */
                fft_prepare_mpi_plans(&r2c_mpi, &c2r_mpi, &mass);
                timer_stop(rank, &run_timer, "Recreating Fourier structures took ");
                message(rank, "\n");
            }
        }

        /* Are we done? */
        if (a_next == a_end)
            continue;

        if (MPI_Rank_Count > 1) {
            message(rank, "Starting particle exchange.\n");
            exchange_particles(particles, boxlen, M, &local_partnum, max_partnum, /* iteration = */ 0, 0, 0, 0, 0);
            timer_stop(rank, &run_timer, "Exchanging particles took ");
        }


        /* Step forward */
        a = a_next;

        message(rank, "\n");
    }

    /* Free the lists with output times */
    free(output_list_snap);
    free(output_list_power);
    free(output_list_halos);
    free(powspec_types);

    /* Free the particle lattice */
    free(particles);

    /* Free the mass grid */
    free_local_grid(&mass);

    /* Destroy FFTW plans */
#ifdef SINGLE_PRECISION_FFTW
    fftwf_destroy_plan(r2c_mpi);
    fftwf_destroy_plan(c2r_mpi);
#else
    fftw_destroy_plan(r2c_mpi);
    fftw_destroy_plan(c2r_mpi);
#endif

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
