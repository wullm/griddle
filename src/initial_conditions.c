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

#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../include/initial_conditions.h"
#include "../include/initial_conditions_ode.h"
#include "../include/gaussian_field.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/grid_io.h"
#include "../include/particle.h"
#include "../include/mesh_grav.h"
#include "../include/fermi_dirac.h"
#include "../include/message.h"
#include "../include/relativity.h"
#include "../include/particle_exchange.h"

int backscale_transfers(struct perturb_data *ptdat, struct cosmology *cosmo,
                        struct cosmology_tables *ctabs, struct units *us,
                        struct physical_consts *pcs, double z_start,
                        double z_target, double *f_asymptotic) {

    /* The number of wavevectors and timesteps in the perturbation data set */
    const int k_size = ptdat->k_size;
    const int tau_size = ptdat->tau_size;

    /* Find the density transfer function index for CDM, baryons, neutrinos */
    int index_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);
    int index_b = findTitle(ptdat->titles, "d_b", ptdat->n_functions);
    int *index_ncdm = malloc(cosmo->N_nu * sizeof(int));
    for (int i = 0; i < cosmo->N_nu; i++) {
        char nu_title[50];
        sprintf(nu_title, "d_ncdm[%d]", i);
        index_ncdm[i] = findTitle(ptdat->titles, nu_title, ptdat->n_functions);
    }

    /* Find the energy flux transfer function index for CDM and baryons */
    int index_t_cdm = findTitle(ptdat->titles, "t_cdm", ptdat->n_functions);
    int index_t_b = findTitle(ptdat->titles, "t_b", ptdat->n_functions);

    /* The baryon fraction of CDM + baryons */
    const double f_b = cosmo->Omega_b / (cosmo->Omega_b + cosmo->Omega_cdm);

    /* Pointer to the density transfer functions */
    double *tf_cdm = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_cdm;
    double *tf_b = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_b;
    double **tf_ncdm = malloc(cosmo->N_nu * sizeof(double*));
    for (int i = 0; i < cosmo->N_nu; i++) {
        tf_ncdm[i] = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_ncdm[i];
    }

    /* Pointer to the energy flux transfer functions */
    double *tf_t_cdm = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_t_cdm;
    double *tf_t_b = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_t_b;

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    struct strooklat spline_k = {ptdat->k, ptdat->k_size};
    init_strooklat_spline(&spline_z, 100);
    init_strooklat_spline(&spline_k, 100);

    /* Scale factors corresponding to the starting and target redshifts */
    const double a_start = 1.0 / (1.0 + z_start);
    const double a_target = 1.0 / (1.0 + z_target);
    /* Redshifts for finite difference around z_start */
    const double log_a_start = log(a_start);
    const double delta_log_a = 0.002;
    const double a_min = exp(log_a_start - delta_log_a);
    const double a_pls = exp(log_a_start + delta_log_a);
    const double z_min = 1.0 / a_min - 1.0;
    const double z_pls = 1.0 / a_pls - 1.0;

    /* Prepare the fluid integrator */
    const double tol = 1e-12;
    const double hstart = 1e-12;
    prepare_fluid_integrator(cosmo, us, pcs, ctabs, tol, hstart);

    /* Allocate memory for neutrino quantities */
    double *d_n_min = malloc(cosmo->N_nu * sizeof(double));
    double *d_n_pls = malloc(cosmo->N_nu * sizeof(double));
    double *d_n = malloc(cosmo->N_nu * sizeof(double));
    double *g_n = malloc(cosmo->N_nu * sizeof(double));
    double *D_n = malloc(cosmo->N_nu * sizeof(double));

    /* Compute the mean asymptotic logarithmic CDM + baryon growth rate */
    double fcb_asymptotic_sum = 0.;
    int count_asymptotic = 0;

    /* Integrate the first-order Newtonian fluid equations */
    for (int i = 0; i < k_size; i++) {
        /* Obtain the transfer functions around z_start by interpolating */
        double d_c_min = strooklat_interp_2d(&spline_z, &spline_k, tf_cdm, z_min, ptdat->k[i]);
        double d_c_pls = strooklat_interp_2d(&spline_z, &spline_k, tf_cdm, z_pls, ptdat->k[i]);
        double d_c_target = strooklat_interp_2d(&spline_z, &spline_k, tf_cdm, z_target, ptdat->k[i]);
        double d_c = strooklat_interp_2d(&spline_z, &spline_k, tf_cdm, z_start, ptdat->k[i]);
        double d_b_min = strooklat_interp_2d(&spline_z, &spline_k, tf_b, z_min, ptdat->k[i]);
        double d_b_pls = strooklat_interp_2d(&spline_z, &spline_k, tf_b, z_pls, ptdat->k[i]);
        double d_b_target = strooklat_interp_2d(&spline_z, &spline_k, tf_b, z_target, ptdat->k[i]);
        double d_b = strooklat_interp_2d(&spline_z, &spline_k, tf_b, z_start, ptdat->k[i]);
        for (int n = 0; n < cosmo->N_nu; n++) {
            d_n_min[n] = strooklat_interp_2d(&spline_z, &spline_k, tf_ncdm[n], z_min, ptdat->k[i]);
            d_n_pls[n] = strooklat_interp_2d(&spline_z, &spline_k, tf_ncdm[n], z_pls, ptdat->k[i]);
            d_n[n] = strooklat_interp_2d(&spline_z, &spline_k, tf_ncdm[n], z_start, ptdat->k[i]);
        }

        /* Compute the logarithmic growth rates (also known as f) */
        double g_c = (d_c_pls - d_c_min) / (2.0 * delta_log_a) / d_c;
        double g_b = (d_b_pls - d_b_min) / (2.0 * delta_log_a) / d_b;
        for (int n = 0; n < cosmo->N_nu; n++) {
            g_n[n] = (d_n_pls[n] - d_n_min[n]) / (2.0 * delta_log_a) / d_n[n];
        }

        /* Compute the weighted CDM + baryon growth rate */
        if (ptdat->k[i] >= 1.0) {
            double d_cb_min = (1.0 - f_b) * d_c_min + f_b * d_b_min;
            double d_cb_pls = (1.0 - f_b) * d_c_pls + f_b * d_b_pls;
            double d_cb = (1.0 - f_b) * d_c + f_b * d_b;
            double g_cb = (d_cb_pls - d_cb_min) / (2.0 * delta_log_a) / d_cb;

            fcb_asymptotic_sum += g_cb;
            count_asymptotic++;
        }

        /* Initialise the input data for the fluid equations */
        struct growth_factors gfac;
        gfac.k = ptdat->k[i];
        gfac.delta_c = d_c;
        gfac.delta_b = d_b;
        gfac.delta_n = d_n;
        gfac.gc = g_c;
        gfac.gb = g_b;
        gfac.gn = g_n;
        gfac.Dc = 0.;
        gfac.Db = 0.;
        gfac.Dn = D_n;

        integrate_fluid_equations(cosmo, us, pcs, ctabs, &gfac, a_start, a_target);

        /* Replace the CDM transfer function by a back-scaled CDM + baryon TF */
        double d_c_scaled = d_c_target * gfac.Dc;
        double d_b_scaled = d_b_target * gfac.Db;
        double d_cb_scaled = (1.0 - f_b) * d_c_scaled + f_b * d_b_scaled;

        /* We replace the entire function (at all redshifts) to the constant
         * value at z_start, just for simplicity */
        for (int j = 0; j < tau_size; j++) {
            tf_cdm[k_size * j + i] = d_cb_scaled;
        }

        /* Obtain the energy flux around z_start by interpolating */
        double t_c = strooklat_interp_2d(&spline_z, &spline_k, tf_t_cdm, z_start, ptdat->k[i]);
        double t_b = strooklat_interp_2d(&spline_z, &spline_k, tf_t_b, z_start, ptdat->k[i]);

        /* We want to use the linear theory growth rate at z = z_start, so
         * we need to keep the ratio theta_i / delta_i */
        double t_c_scaled = t_c / d_c * d_c_scaled;
        double t_b_scaled = t_b / d_b * d_b_scaled;
        double t_cb_scaled = (1.0 - f_b) * t_c_scaled + f_b * t_b_scaled;

        /* Replace the the CDM energy flux transfer function by the scaled
         * CDM + baryon transfer function. */
        for (int j = 0; j < tau_size; j++) {
            tf_t_cdm[k_size * j + i] = t_cb_scaled;
        }
    }

    /* The mean asymptotic growth rate */
    *f_asymptotic = fcb_asymptotic_sum / count_asymptotic;

    /* Free memory for neutrino quantities */
    free(d_n_min);
    free(d_n_pls);
    free(d_n);
    free(g_n);
    free(D_n);
    free(tf_ncdm);
    free(index_ncdm);

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    free_strooklat_spline(&spline_k);

    return 0;
}

int generate_potential_grid(struct distributed_grid *dgrid, long int Seed,
                            char fix_modes, char invert_modes,
                            enum potential_type type,
                            struct perturb_data *ptdat,
                            struct cosmology *cosmo, double z_start) {

    /* Find the transfer function index for CDM */
    int index_cdm;
    if (type == density_potential_type) {
        index_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);
    } else if (type == velocity_potential_type) {
        index_cdm = findTitle(ptdat->titles, "t_cdm", ptdat->n_functions);
    } else {
        printf("Error: potential type not implemented.\n");
        exit(1);
    }

    /* Generate a complex Hermitian Gaussian random field */
    generate_ngeniclike_grf(dgrid, Seed);
    enforce_hermiticity(dgrid);

    /* Apply fixing and/or inverting of the modes for variance reduction? */
    if (fix_modes || invert_modes) {
        fix_and_pairing(dgrid, fix_modes, invert_modes);
    }

    /* Apply the primordial power spectrum without transfer functions */
    fft_apply_kernel_dg(dgrid, dgrid, kernel_power_no_transfer, cosmo);

    /* Pointer to the CDM density transfer function */
    double *tf = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_cdm;

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    struct strooklat spline_k = {ptdat->k, ptdat->k_size};
    init_strooklat_spline(&spline_z, 100);
    init_strooklat_spline(&spline_k, 100);

    /* Apply the CDM density transfer function */
    struct spline_params sp = {&spline_z, &spline_k, /* z = */ z_start, tf};
    fft_apply_kernel_dg(dgrid, dgrid, kernel_transfer_function, &sp);

    /* Compute the potential by applying the inverse Poisosn kernel */
    fft_apply_kernel_dg(dgrid, dgrid, kernel_inv_poisson_alt, NULL);

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    free_strooklat_spline(&spline_k);

    /* Execute the Fourier transform and normalize */
    fft_c2r_dg(dgrid);

    /* Export the GRF */
    // writeFieldFile_dg(dgrid, "grid.hdf5");

    return 0;
}

/* Compute the 2LPT potential. This requires two additional working memory
   grids to be allocated. */
int generate_2lpt_grid(struct distributed_grid *dgrid,
                       struct distributed_grid *temp1,
                       struct distributed_grid *temp2,
                       struct distributed_grid *dgrid_2lpt,
                       struct perturb_data *ptdat,
                       struct cosmology *cosmo, double z_start) {

    /* Execute the Fourier transform and normalize */
    fft_r2c_dg(dgrid);

    /* We calculate derivatives using FFT kernels */
    const kernel_func derivatives[] = {kernel_dx, kernel_dy, kernel_dz};

    /* Erase the current data */
    for (long int i = 0; i < dgrid_2lpt->local_real_size; i++) {
        dgrid_2lpt->box[i] = 0.;
    }

    /* First add the (xx)*(yy) + (xx)*(zz) + (yy)*(zz) terms */

    /* We need xy, xz, yz to compute the Hessian */
    const int index_a[] = {0, 0, 1};
    const int index_b[] = {1, 2, 2};

    /* Compute the 3 derivative components of the Hessian */
    for (int j=0; j<3; j++) {
        /* Compute the derivative d^2 phi / (dx_i dx_j) */
        fft_apply_kernel_dg(temp1, dgrid, derivatives[index_a[j]], NULL);
        fft_apply_kernel_dg(temp1, temp1, derivatives[index_a[j]], NULL);

        /* Compute the derivative d^2 phi / (dx_i dx_j) */
        fft_apply_kernel_dg(temp2, dgrid, derivatives[index_b[j]], NULL);
        fft_apply_kernel_dg(temp2, temp2, derivatives[index_b[j]], NULL);

        /* Fourier transform to configuration space */
        fft_c2r_dg(temp1);
        fft_c2r_dg(temp2);

        /* Add the product (=convolution) to the intermediate answer */
        for (long int i = 0; i < dgrid_2lpt->local_real_size; i++) {
            dgrid_2lpt->box[i] += temp1->box[i] * temp1->box[i];
        }
    }

    /* Now add the (xy)*(xy) + (xz)*(xz) + (yz)*(yz) terms */

    /* Compute the 3 derivative components of the Hessian */
    for (int j=0; j<3; j++) {
        /* Compute the derivative d^2 phi / (dx_i dx_j) */
        fft_apply_kernel_dg(temp1, dgrid, derivatives[index_a[j]], NULL);
        fft_apply_kernel_dg(temp1, temp1, derivatives[index_b[j]], NULL);

        /* Fourier transform to configuration space */
        fft_c2r_dg(temp1);

        /* Subtract the square (=convolution) from the intermediate answer */
        for (long int i = 0; i < dgrid_2lpt->local_real_size; i++) {
            dgrid_2lpt->box[i] -= temp1->box[i] * temp1->box[i];
        }
    }

    /* Fourier transform to momentum space */
    fft_r2c_dg(dgrid_2lpt);

    /* Compute the potential by applying the inverse Poisosn kernel */
    fft_apply_kernel_dg(dgrid_2lpt, dgrid_2lpt, kernel_inv_poisson_alt, NULL);

    /* Divide the potential by two */
    double factor = 0.5;
    fft_apply_kernel_dg(dgrid_2lpt, dgrid_2lpt, kernel_constant, &factor);

    /* Fourier transform to configuration space */
    fft_c2r_dg(dgrid_2lpt);
    fft_c2r_dg(dgrid);

    /* Export the 2LPT potential grid */
    // writeFieldFile_dg(dgrid_2lpt, "2lpt.hdf5");

    return 0;
}

int generate_particle_lattice(struct distributed_grid *lpt_potential,
                              struct distributed_grid *lpt_potential_2,
                              struct distributed_grid *velocity_potential,
                              struct perturb_data *ptdat,
                              struct particle *parts, struct cosmology *cosmo,
                              struct units *us, struct physical_consts *pcs,
                              long long particle_offset, long long X0,
                              long long NX, double z_start,
                              double f_asymptotic) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    init_strooklat_spline(&spline_z, 100);

    /* The logarithmic growth rate */
    double f_start;
    if (f_asymptotic > 0.) {
        f_start = f_asymptotic;
    } else {
        f_start = strooklat_interp(&spline_z, ptdat->f_growth, z_start);
    }

    /* The velocity factor aHf */
    const double a_start = 1.0 / (1.0 + z_start);
    const double H_start = strooklat_interp(&spline_z, ptdat->Hubble_H, z_start);
    const double vel_fact = a_start * f_start * H_start;

    /* The 2LPT neutrino correction factor (2202.00670) */
    double f_nu_tot_0 = 0.;
    for (int i = 0; i < cosmo->N_nu; i++) {
        f_nu_tot_0 += cosmo->f_nu_0[i];
    }
    const double factor_nu_2lpt = 1.0 + (4. / 35.) * f_nu_tot_0;

    /* The 2LPT factor */
    const double factor_2lpt = 3. / 7. * factor_nu_2lpt;
    const double factor_vel_2lpt = factor_2lpt * 2.0;

    /* Grid constants */
    const int N = lpt_potential->N;
    const double boxlen = lpt_potential->boxlen;

    /* Cosmological constants */
#ifdef WITH_MASSES
    const double h = cosmo->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
    const double Omega_cb = cosmo->Omega_cdm + cosmo->Omega_b;
    const double part_mass = rho_crit * Omega_cb * pow(boxlen / N, 3);
#endif

    /* Position factors */
    const double grid_fac = boxlen / N;
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;

    /* Generate a particle lattice */
    for (int i = X0; i < X0 + NX; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                struct particle *part = &parts[particle_offset + (i - X0) * N * N + j * N + k];
#ifdef WITH_PARTICLE_IDS
                part->id = (long long int) i * N * N + j * N + k;
#endif

                /* Regular grid positions */
                double x[3] = {i, j, k};

                /* Zel'dovich displacement */
                double dx[3] = {0,0,0};
                accelCIC(lpt_potential, x, dx);

                /* The 2LPT displacement */
                double dx2[3] = {0,0,0};
                accelCIC(lpt_potential_2, x, dx2);

                /* The linear theory velocity (~ dx + relativistc corrections) */
                double v[3] = {0,0,0};
                accelCIC(velocity_potential, x, v);

                /* CDM */
#ifdef WITH_PARTTYPE
                part->type = 1;
#endif

                /* Add the displacements */
                x[0] = x[0] * grid_fac - (dx[0] + factor_2lpt * dx2[0]);
                x[1] = x[1] * grid_fac - (dx[1] + factor_2lpt * dx2[1]);
                x[2] = x[2] * grid_fac - (dx[2] + factor_2lpt * dx2[2]);

                /* Convert positions to integers */
                part->x[0] = pos_to_int_fac * x[0];
                part->x[1] = pos_to_int_fac * x[1];
                part->x[2] = pos_to_int_fac * x[2];

                /* Set the velocities */
                part->v[0] = a_start * (v[0] - vel_fact * factor_vel_2lpt * dx2[0]);
                part->v[1] = a_start * (v[1] - vel_fact * factor_vel_2lpt * dx2[1]);
                part->v[2] = a_start * (v[2] - vel_fact * factor_vel_2lpt * dx2[2]);

#ifdef WITH_MASSES
                part->m = part_mass;
#endif
            }
        }
    }

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);

    return 0;
}

int generate_neutrinos(struct particle *parts, struct cosmology *cosmo,
                       struct cosmology_tables *ctabs, struct units *us,
                       struct physical_consts *pcs, long long int N_nupart,
                       long long particle_offset, long long local_cdm_num,
                       long long local_neutrino_num, double boxlen,
                       long long X0_nupart, long long NX_nupart,
                       double z_start, rng_state *state) {

    /* Create interpolation splines for scale factors */
    struct strooklat spline_a = {ctabs->avec, ctabs->size};
    init_strooklat_spline(&spline_a, 100);

    /* Cosmological constants */
#ifdef WITH_MASSES
    const double h = cosmo->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
    const double base_part_mass = rho_crit * pow(boxlen / N_nupart, 3);
#endif

    /* Pull down the present day neutrino density per species */
    double *Omega_nu_0 = malloc(cosmo->N_nu * sizeof(double));
    for (int i = 0; i < cosmo->N_nu; i++) {
        Omega_nu_0[i] = strooklat_interp(&spline_a, ctabs->Omega_nu + i * ctabs->size, 1.0);
    }

    /* Fermi-Dirac conversion factor kb*T to km/s */
    const double fac = (pcs->SpeedOfLight * cosmo->T_nu_0 * pcs->kBoltzmann) / pcs->ElectronVolt;

    /* Position factors */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;

    /* Generate neutrinos */
    for (long long i = 0; i < local_neutrino_num; i++) {
        struct particle *part = &parts[particle_offset + i];
        long int seed_and_id = local_cdm_num + i;
#ifdef WITH_PARTICLE_IDS
        part->id = seed_and_id;
#endif

        /* Neutrino */
#ifdef WITH_PARTTYPE
        part->type = 6;

        /* Initially the weight is zero */
        part->w = 0.;
#endif

        /* Sample a position uniformly in the box (on this rank) */
        part->x[0] = pos_to_int_fac * (sampleUniform(state) * NX_nupart + X0_nupart) * boxlen / N_nupart;
        part->x[1] = pos_to_int_fac * sampleUniform(state) * boxlen;
        part->x[2] = pos_to_int_fac * sampleUniform(state) * boxlen;

        /* Sample a neutrino species */
        int species = (int)(seed_and_id % cosmo->N_nu);
        double m_eV = cosmo->M_nu[species];
#ifdef WITH_MASSES
        part->m = Omega_nu_0[species] * base_part_mass;
#endif

        /* Sample a deterministic Fermi-Dirac momentum */
        double n[3];
        double p = neutrino_seed_to_fermi_dirac(seed_and_id) * fac / m_eV;
        neutrino_seed_to_direction(seed_and_id, n);

        part->v[0] = p * n[0];
        part->v[1] = p * n[1];
        part->v[2] = p * n[2];
    }

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_a);
    /* Free the neutrino density array */
    free(Omega_nu_0);

    return 0;
}

int pre_integrate_neutrinos(struct distributed_grid *dgrid, struct perturb_data *ptdat,
                            struct params *pars, char fix_modes, char invert_modes,
                            struct particle *parts, struct cosmology *cosmo,
                            struct cosmology_tables *ctabs, struct units *us,
                            struct physical_consts *pcs, long long int N_nupart,
                            long long local_partnum, long long max_partnum,
                            long long local_neutrino_num, double boxlen,
                            double z_start, long int Seed) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Find the transfer function index for the gravitational potential */
    int index_phi = findTitle(ptdat->titles, "phi", ptdat->n_functions);

    /* Pointer to the transfer function */
    double *tf = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_phi;

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    struct strooklat spline_k = {ptdat->k, ptdat->k_size};
    init_strooklat_spline(&spline_z, 100);
    init_strooklat_spline(&spline_k, 100);

    /* Additional interpolation spline for the background cosmology scale factors */
    struct strooklat spline_bg_a = {ctabs->avec, ctabs->size};
    init_strooklat_spline(&spline_bg_a, 100);

    /* Before starting the main simulation, we will integrate the neutrinos
     * from some early time down to the starting redshift of the simulation. */
    const double a_factor = 1.0 + pars->NeutrinoScaleFactorEarlyStep;
    const double a_start = 1.0 / (1.0 + z_start);
    const double a_early = pars->NeutrinoScaleFactorEarly;
    const double a_stop = a_start;
    const int MAX_ITER = (log(a_stop) - log(a_early))/log(a_factor) + 1;
    double a = a_early;

    if (a_early < ctabs->avec[0] || a_early < 1.0 / (1.0 + ptdat->redshift[0])) {
        printf("Error: Neutrino:ScaleFactorEarly smaller than earliest time in "
               "cosmology or linear perturbation tables (%g < %g or %g < %g).\n",
               a_early, ctabs->avec[0], a_early, 1.0 / (1.0 + ptdat->redshift[0]));
        exit(1);
    }

    if (a_early > a_start) {
        printf("Error: Neutrino:ScaleFactorEarly larger than the starting scale "
               "factor of the simulation.\n");
        exit(1);
    }

    /* Pointer to and dimensions of the real-space potential grid */
    const GridFloatType *box = dgrid->buffered_box;
    const long int M = dgrid->N;
    const long int Mz = dgrid->Nz;
    const long int MX0 = dgrid->X0;
    const long int buffer_width = dgrid->buffer_width;

    /* Conversion factor between floating point and integer positions */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;

    /* Position conversion factors to [0, M] where M is the grid size */
    const double grid_to_int_fac = pow(2.0, POSITION_BITS) / M;
    const double int_to_grid_fac = 1.0 / grid_to_int_fac;
    const double cell_fac = M / boxlen;

    /* The neutrino pre-integration loop */
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
            a_next = a_stop;
        }

        /* Compute the current redshift and log conformal time */
        double z = 1./a - 1.;

        /* Determine the adjacent half-step scale factor */
        double a_half_prev = sqrt(a_prev * a);
        double a_half_next = sqrt(a_next * a);

        /* Obtain the kick and drift time steps */
        double kick_dtau  = strooklat_interp(&spline_bg_a, ctabs->kick_factors, a_half_next) -
                            strooklat_interp(&spline_bg_a, ctabs->kick_factors, a_half_prev);
        double drift_dtau = strooklat_interp(&spline_bg_a, ctabs->drift_factors, a_next) -
                            strooklat_interp(&spline_bg_a, ctabs->drift_factors, a);

        message(rank, "Neutrino pre-integration step %d at z = %g\n", ITER, z);

        /* Generate a complex Hermitian Gaussian random field */
        // generate_complex_grf(dgrid, seed);
        generate_ngeniclike_grf(dgrid, Seed);
        enforce_hermiticity(dgrid);

        /* Apply fixing and/or inverting of the modes for variance reduction? */
        if (fix_modes || invert_modes) {
            fix_and_pairing(dgrid, fix_modes, invert_modes);
        }

        /* Apply the primordial power spectrum without transfer functions */
        fft_apply_kernel_dg(dgrid, dgrid, kernel_power_no_transfer, cosmo);

        /* Apply the potential transfer function */
        struct spline_params sp = {&spline_z, &spline_k, z, tf};
        fft_apply_kernel_dg(dgrid, dgrid, kernel_transfer_function, &sp);

        /* Multiply by a constant factor */
        for (long int i = 0; i < dgrid->local_complex_size; i++) {
            dgrid->fbox[i] *= - a;
        }

        /* Execute the Fourier transform and normalize */
        fft_c2r_dg(dgrid);

        /* Create buffers for the potential */
        create_local_buffers(dgrid);

        /* Integrate the neutrino particles */
        for (long long i = 0; i < local_partnum; i++) {
            struct particle *p = &parts[i];

#ifdef WITH_PARTTYPE
            /* Only integrate neutrinos */
            if (p->type != 6) continue;
#endif

            /* Convert integer positions to floating points on [0, M] */
            double x[3] = {p->x[0] * int_to_grid_fac,
                           p->x[1] * int_to_grid_fac,
                           p->x[2] * int_to_grid_fac};

            /* Obtain the acceleration by differentiating the potential */
            double acc[3] = {0, 0, 0};
            accelCIC_2nd(box, x, acc, M, MX0, buffer_width, Mz, cell_fac);

            /* Execute kick */
            FloatVelType v[3] = {p->v[0] + acc[0] * kick_dtau,
                                 p->v[1] + acc[1] * kick_dtau,
                                 p->v[2] + acc[2] * kick_dtau};

            /* Relativistic drift correction */
            const double rel_drift = relativistic_drift(v, p, pcs, a);

            /* Execute drift */
            p->x[0] += v[0] * drift_dtau * rel_drift * pos_to_int_fac;
            p->x[1] += v[1] * drift_dtau * rel_drift * pos_to_int_fac;
            p->x[2] += v[2] * drift_dtau * rel_drift * pos_to_int_fac;

            /* Update velocities */
            p->v[0] = v[0];
            p->v[1] = v[1];
            p->v[2] = v[2];
        }

        /* Exchange partciles between MPI tasks */
        if (MPI_Rank_Count > 1) {
            exchange_particles(parts, boxlen, M, &local_partnum, max_partnum, /* iteration = */ 0, 0, 0, 0, 0);
        }

        /* Step forward */
        a = a_next;
    }

#if defined(WITH_PARTTYPE) && defined(WITH_PARTICLE_IDS)
    /* Conversion factor for neutrino momenta */
    const double neutrino_qfac = pcs->ElectronVolt / (pcs->SpeedOfLight * cosmo->T_nu_0 * pcs->kBoltzmann);

    /* Finally, set the delta-f weights of the neutrino particles */
    for (long long i = 0; i < local_partnum; i++) {
        struct particle *p = &parts[i];

        if (p->type == 6) {
            double m_eV = cosmo->M_nu[(int)p->id % cosmo->N_nu];
            double v2 = p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2];
            double q = sqrt(v2) * neutrino_qfac * m_eV;
            double qi = neutrino_seed_to_fermi_dirac(p->id);
            double f = fermi_dirac_density(q);
            double fi = fermi_dirac_density(qi);

            p->w = 1.0 - f / fi;
        }
    }
#endif

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    free_strooklat_spline(&spline_k);

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_bg_a);

    return 0;
}