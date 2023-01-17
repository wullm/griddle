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

#ifndef MASS_DEPOSIT_H
#define MASS_DEPOSIT_H

#include "../include/distributed_grid.h"
#include "../include/particle.h"
#include "../include/units.h"
#include "../include/fft.h"
#include "../include/cosmology.h"

enum grid_type {
    all_mass,
    cb_mass,
    nu_mass,
    num_grid_types
};

extern const char *grid_type_names[num_grid_types];

int mass_deposition(struct distributed_grid *dgrid, struct particle *parts,
                    long long int local_partnum, enum grid_type gtype,
                    const struct cosmology *cosmo,
                    const struct physical_consts *pcs);
int compute_potential(struct distributed_grid *dgrid,
                      struct physical_consts *pcs, FourierPlanType r2c,
                      FourierPlanType c2r);

#endif
