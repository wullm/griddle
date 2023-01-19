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
    CellOffsetIntType offset;
    CellIntType cell;
};

struct so_part_data {
    float m;
    float r;
    float Delta;
};

struct so_halo {
    /* Global ID of the halo (matches the corresponding FOF halo) */
    long int global_id;
    /* Shrinking sphere centre of mass (dark matter particles up to R_inner) */
    double x_com_inner[3];
    /* Shrinking sphere centre of mass velocity */
    double v_com_inner[3];
    /* Centre of mass of all SO particles */
    double x_com[3];
    /* Centre of mass velocity of all SO particles */
    double v_com[3];
    /* Total mass of the SO particles */
    double mass_tot; // ( = M_SO up to errors)
    /* Spherical overdensity mass, given by (4/3) pi R_SO^3 Delta rho_crit */
    double M_SO;
    /* Spherical overdensity radius */
    double R_SO;
    /* Radius enclosing the innermost particles that determine the CoM */
    double R_inner;
    /* Total mass of dark matter SO particles */
    double mass_dm;
    /* Total mass of neutrino SO particles */
    double mass_nu;
    /* Centre of mass of all dark matter SO particles */
    double x_com_dm[3];
    /* Centre of mass velocity of all dark matter SO particles */
    double v_com_dm[3];
    /* The maximum circular velocity */
    double v_max;
    /* The radius at the maximum circular velocity */
    double R_v_max;
    /* The half-mass radius */
    double R_half_mass;
    /* Total number of particles within the SO radius */
    int npart_tot;
    /* Total number of dark matter particles within the SO radius */
    int npart_dm;
    /* Total number of neutrino particles within the SO radius */
    int npart_nu;
    /* Home rank of the halo */
    int rank;
};

static inline int soPartSort(const void *a, const void *b) {
    struct so_part_data *pa = (struct so_part_data*) a;
    struct so_part_data *pb = (struct so_part_data*) b;

    return pa->r >= pb->r;
}

int find_overlapping_cells(const double com[3], double search_radius,
                           double pos_to_cell_fac, CellIntType N_cells,
                           CellIntType **cells, CellIntType *num_overlap);
int analysis_so(struct particle *parts, struct fof_halo **fofs, double boxlen,
                long int N_cb, long int N_nu, long long int Ng,
                long long int num_localpart, long long int max_partnum,
                long int total_num_fofs, long int num_local_fofs,
                int output_num, double a_scale_factor, const struct units *us,
                const struct physical_consts *pcs,
                const struct cosmology *cosmo, const struct params *pars,
                const struct cosmology_tables *ctabs,
                double dtau_kick, double dtau_drift);

#endif
