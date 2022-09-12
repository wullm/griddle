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

#ifndef ICS_H
#define ICS_H

#include "../include/random.h"
#include "../include/perturb_data.h"
#include "../include/cosmology.h"
#include "../include/distributed_grid.h"
#include "../include/particle.h"

int generate_potential_grid(struct distributed_grid *dgrid, rng_state *seed,
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
                              struct perturb_params *ptpars,
                              struct particle *parts, struct cosmology *cosmo,
                              struct units *us, struct physical_consts *pcs,
                              double z_start);
#endif
