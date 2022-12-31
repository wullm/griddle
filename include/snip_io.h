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

/* Methods for output of particle snipshots. "Snip"-shots are reduced versions
 * of particle snapshots, containing only some fraction of particles contained
 * in halos. */

#ifndef SNIP_IO_H
#define SNIP_IO_H

#include <hdf5.h>

#include "particle.h"
#include "params.h"
#include "units.h"
#include "analysis_so.h"

int exportSnipshot(const struct params *pars, const struct units *us,
                   const struct so_halo *halos, const struct physical_consts *pcs,
                   const struct particle *parts, const struct cosmology *cosmo,
                   const struct so_cell_list *cell_list, long int *cell_counts,
                   long int *cell_offsets, int output_num, double a_scale_factor,
                   long int N_cells, double reduce_factor,
                   int min_part_export_per_halo, long long int local_partnum,
                   long long int local_halo_num);

#endif
