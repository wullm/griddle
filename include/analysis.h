/*******************************************************************************
 * This file is part of Sedulus.
 * Copyright (c) 2023 Willem Elbers (whe@willemelbers.com)
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

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "particle.h"
#include "units.h"

IntPosType position_checksum(const struct particle *parts, long int local_partnum);
int drift_particles(struct particle *parts, long int local_partnum,
                    double a, double drift_dtau, double pos_to_int_fac,
                    const struct physical_consts *pcs);

#endif
