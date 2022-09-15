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

/* Methods for input and output of particle snapshots */

#ifndef SNAP_IO_H
#define SNAP_IO_H

#include <hdf5.h>

#include "particle.h"
#include "params.h"
#include "units.h"

int exportSnapshot(struct params *pars, struct units *us,
                   struct particle *particles, int output_num, double a,
                   int N, long long int local_partnum);
int writeHeaderAttributes(struct params *pars, struct units *us, double a,
                          long long int *numparts_local, long long int *numparts_total,
                          hid_t h_file);
int readSnapshot(struct params *pars, struct units *us,
                 struct particle *particles, const char *fname, double a,
                 long long int local_partnum, long long int local_firstpart,
                 long long int max_partnum);

#endif
