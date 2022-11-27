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

/* On-the-fly spherical overdensity halo finder */

#ifndef ANALYSIS_SO_H
#define ANALYSIS_SO_H

#include "analysis_fof.h"
#include "particle.h"
#include "units.h"
#include "cosmology.h"

struct so_cell_list {
    long int offset;
    int cell;
};

struct so_halo {
    long int global_id;
    double x_com[3];
    double v_com[3];
    double mass_tot; // mass of particles = (M_SO up to errors)
    double M_SO; // mass
    double R_SO; // radius
    int npart_tot; // number of particles within SO radius
    int rank;
};

int analysis_so(struct particle *parts, struct fof_halo *fofs, double boxlen,
                long int Np, long long int Ng, long long int num_localpart,
                long long int max_partnum, long int num_fofs, int output_num,
                double a_scale_factor, const struct units *us,
                const struct physical_consts *pcs,
                const struct cosmology *cosmo);

#endif
