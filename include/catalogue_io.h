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

/* Methods for input and output of halo catalogues */

#ifndef CATALOGUE_IO_H
#define CATALOGUE_IO_H

#include <hdf5.h>

#include "particle.h"
#include "params.h"
#include "units.h"
#include "analysis_fof.h"

int exportCatalogue(const struct params *pars, const struct units *us,
                    const struct physical_consts *pcs, int output_num, double a,
                    long int total_num_structures, long int local_num_structures,
                    struct fof_halo *fofs);

#endif
