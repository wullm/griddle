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

int generate_potential_grid(struct distributed_grid *dgrid, rng_state *seed,
                            struct perturb_data *ptdat,
                            struct cosmology *cosmo, double z_start) {

    /* Find the transfer function index for CDM */
    int index_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);
    
    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(dgrid, seed);
    enforce_hermiticity(dgrid);
    
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
    fft_apply_kernel_dg(dgrid, dgrid, kernel_inv_poisson, NULL);
    
    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    free_strooklat_spline(&spline_k);
        
    /* Execute the Fourier transform and normalize */
    fft_c2r_dg(dgrid);
    
    /* Export the GRF */
    // writeFieldFile_dg(dgrid, "grid.hdf5");
    
    return 0;
}

int generate_particle_lattice(struct distributed_grid *lpt_potential, 
                              struct perturb_data *ptdat,
                              struct perturb_params *ptpars,
                              struct particle *parts, struct cosmology *cosmo,
                              struct units *us, struct physical_consts *pcs,
                              double z_start) {

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    init_strooklat_spline(&spline_z, 100);
    
    /* The velocity factor aHf */
    const double a_start = 1.0 / (1.0 + z_start);
    const double f_start = strooklat_interp(&spline_z, ptdat->f_growth, z_start);
    const double H_start = strooklat_interp(&spline_z, ptdat->Hubble_H, z_start);
    const double vel_fact = a_start * f_start * H_start;
    
    /* Grid constants */
    const int N = lpt_potential->N;
    const double boxlen = lpt_potential->boxlen;
    
    /* Cosmological constants */
    const double h = cosmo->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
    const double Omega_m = ptpars->Omega_m;
    const double part_mass = rho_crit * Omega_m * pow(boxlen / N, 3);

    /* Generate a particle lattice */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                struct particle *part = &parts[i * N * N + j * N + k];
                part->id = (long long int) i * N * N + j * N + k;
                
                part->x[0] = i * boxlen / N;
                part->x[1] = j * boxlen / N;
                part->x[2] = k * boxlen / N;                
                
                double dx[3] = {0,0,0};
                
                accelCIC(lpt_potential, N, boxlen, part->x, dx);
                                
                part->x[0] -= dx[0];
                part->x[1] -= dx[1];
                part->x[2] -= dx[2];
                
                part->v[0] = -vel_fact * dx[0];
                part->v[1] = -vel_fact * dx[1];
                part->v[2] = -vel_fact * dx[2];
                part->m = part_mass;
            }
        }
    }
    
    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    
    return 0;
}