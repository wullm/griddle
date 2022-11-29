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
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../include/initial_conditions.h"
#include "../include/gaussian_field.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/grid_io.h"
#include "../include/particle.h"
#include "../include/mesh_grav.h"
#include "../include/fermi_dirac.h"

int generate_potential_grid(struct distributed_grid *dgrid, rng_state *seed,
                            char fix_modes, char invert_modes,
                            struct perturb_data *ptdat,
                            struct cosmology *cosmo, double z_start) {

    /* Find the transfer function index for CDM */
    int index_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);

    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(dgrid, seed);
    enforce_hermiticity(dgrid);

    /* Apply fixing and/or inverting of the modes for variance reduction? */
    if (fix_modes || invert_modes) {
        fix_and_pairing(dgrid, fix_modes, invert_modes);
    }

    /* Execute the Fourier transform and normalize */
    fft_c2r_dg(dgrid);

    /* Make the grid yz-symmetric */
    for (int i = dgrid->X0; i < dgrid->X0 + dgrid->NX; i++) {
        for (int j = 0; j < dgrid->N; j++) {
            for (int k = j; k < dgrid->N; k++) {
                *point_row_major_dg_buffered(i, j, k, dgrid) = *point_row_major_dg_buffered(i, k, j, dgrid);
            }
        }
    }

    /* Execute the Fourier transform and normalize */
    fft_r2c_dg(dgrid);

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
                              struct perturb_data *ptdat,
                              struct perturb_params *ptpars,
                              struct particle *parts, struct cosmology *cosmo,
                              struct units *us, struct physical_consts *pcs,
                              long long X0, long long NX, double z_start,
                              long long int *local_partnum) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    init_strooklat_spline(&spline_z, 100);

    /* The velocity factor a^2Hf (a^2 for the internal velocity variable) */
    const double a_start = 1.0 / (1.0 + z_start);
    const double f_start = strooklat_interp(&spline_z, ptdat->f_growth, z_start);
    const double H_start = strooklat_interp(&spline_z, ptdat->Hubble_H, z_start);
    const double vel_fact = a_start * a_start * f_start * H_start;

    /* The 2LPT factor */
    const double factor_2lpt = 3. / 7.;
    const double factor_vel_2lpt = factor_2lpt * 2.0;

    /* Grid constants */
    const int N = lpt_potential->N;
    const double boxlen = lpt_potential->boxlen;

    /* Cosmological constants */
#ifdef WITH_MASSES
    const double h = cosmo->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
    const double Omega_m = ptpars->Omega_m;
    const double part_mass = rho_crit * Omega_m * pow(boxlen / N, 3);
#endif

    /* Position factors */
    const double grid_fac = boxlen / N;
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;

    /* Generate a particle lattice */
    long int pcount = 0;
    for (int i = X0; i < X0+NX; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < j; k++) { // only the lower triangle in the yz-plane
                struct particle *part = &parts[pcount];
#ifdef WITH_PARTICLE_IDS
                part->id = pcount;
#endif

                /* Regular grid positions */
                double x[3] = {i, j, k};

                /* Zel'dovich displacement */
                double dx[3] = {0,0,0};
                accelCIC(lpt_potential, x, dx);

                /* The 2LPT displacement */
                double dx2[3] = {0,0,0};
                accelCIC(lpt_potential_2, x, dx2);

                /* CDM */
#ifdef WITH_PARTTYPE
                part->type = 1;
#endif

                /* Add the displacements */
                x[0] -=  dx[0] + factor_2lpt * dx2[0];
                x[1] -=  dx[1] + factor_2lpt * dx2[1];
                x[2] -=  dx[2] + factor_2lpt * dx2[2];

                /* Convert positions to integers */
                part->x[0] = grid_fac * pos_to_int_fac * x[0];
                part->x[1] = grid_fac * pos_to_int_fac * x[1];
                part->x[2] = grid_fac * pos_to_int_fac * x[2];

                if (part->x[2] > part->x[1]) {
                    IntPosType swap = part->x[2];

                    part->x[2] = part->x[1];
                    part->x[1] = swap;
                }

                /* Set the velocities */
                part->v[0] -= vel_fact * (dx[0] + factor_vel_2lpt * dx2[0]);
                part->v[1] -= vel_fact * (dx[1] + factor_vel_2lpt * dx2[1]);
                part->v[2] -= vel_fact * (dx[2] + factor_vel_2lpt * dx2[2]);
#ifdef WITH_MASSES
                part->m = part_mass;
#endif
                pcount++;
            }
        }
    }

    /* Update the local particle number */
    *local_partnum = pcount;

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);

    return 0;
}

int generate_neutrinos(struct particle *parts, struct cosmology *cosmo,
                       struct cosmology_tables *ctabs, struct units *us,
                       struct physical_consts *pcs, long long int N_nupart,
                       long long local_partnum, long long local_neutrino_num,
                       double boxlen, long long X0, long long NX, long long N,
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
        struct particle *part = &parts[local_partnum + i];
        long int seed_and_id = local_partnum + i;
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
        part->x[0] = pos_to_int_fac * (sampleUniform(state) * NX + X0) * boxlen / N;
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
