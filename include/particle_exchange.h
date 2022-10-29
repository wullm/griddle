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

/* Methods for generating Gaussian random fields */

#ifndef PARTICLE_EXCHANGE_H
#define PARTICLE_EXCHANGE_H

#include "particle.h"

int exchange_particles(struct particle *parts, double boxlen, long long int Ng,
                       long long int *num_localpart, long long int max_partnum,
                       int iteration,
                       long long int received_left,
                       long long int received_right);

#endif
