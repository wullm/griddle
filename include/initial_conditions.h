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

#ifndef ICS_H
#define ICS_H

#include "../include/random.h"
#include "../include/perturb_data.h"
#include "../include/cosmology.h"
#include "../include/distributed_grid.h"
#include "../include/particle.h"

int generate_potential_grid(struct distributed_grid *dgrid, long int Seed,
                            char fix_modes, char invert_modes,
                            struct perturb_data *ptdat,
                            struct cosmology *cosmo, double z_start);
int generate_2lpt_grid(struct distributed_grid *dgrid,
                       struct distributed_grid *temp1,
                       struct distributed_grid *temp2,
                       struct distributed_grid *dgrid_2lpt,
                       struct perturb_data *ptdat,
                       struct cosmology *cosmo, double z_start);
int generate_particle_lattice(struct distributed_grid *lpt_potential,
                              struct distributed_grid *lpt_potential_2,
                              struct perturb_data *ptdat,
                              struct particle *parts, struct cosmology *cosmo,
                              struct units *us, struct physical_consts *pcs,
                              long long particle_offset, long long X0,
                              long long NX, double z_start,
                              double f_asymptotic);
int generate_neutrinos(struct particle *parts, struct cosmology *cosmo,
                       struct cosmology_tables *ctabs, struct units *us,
                       struct physical_consts *pcs, long long int N_nupart,
                       long long particle_offset, long long local_cdm_num,
                       long long local_neutrino_num, double boxlen,
                       long long X0_nupart, long long NX_nupart,
                       double z_start, rng_state *state);
int backscale_transfers(struct perturb_data *ptdat, struct cosmology *cosmo,
                        struct cosmology_tables *ctabs, struct units *us,
                        struct physical_consts *pcs, double z_start,
                        double z_target, double *f_asymptotic);
int pre_integrate_neutrinos(struct distributed_grid *dgrid, struct perturb_data *ptdat,
                            struct params *pars, char fix_modes, char invert_modes,
                            struct particle *parts, struct cosmology *cosmo,
                            struct cosmology_tables *ctabs, struct units *us,
                            struct physical_consts *pcs, long long int N_nupart,
                            long long local_partnum, long long max_partnum,
                            long long local_neutrino_num, double boxlen,
                            double z_start, long int Seed);
#endif
