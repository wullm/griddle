/*******************************************************************************
 * This file is part of Nyver.
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

/* Methods for input and output of data cubes, using the HDF5 format */

#ifndef GRIDIO_H
#define GRIDIO_H

#include "distributed_grid.h"

int readFieldFile(double **box, int *N, double *box_len, const char *fname);
int writeFieldFile(const double *box, int N, double boxlen, const char *fname);
int writeFieldFileCompressed(const double *box, int N, double boxlen,
                             const char *fname, int digits);
int writeFieldFile_dg(struct distributed_grid *dg, const char *fname);
#endif
