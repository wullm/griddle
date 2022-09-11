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

#include <stdlib.h>
#include <math.h>

#include "../include/cosmology.h"

/* The .ini parser library is minIni */
#include "../parser/minIni.h"

int readCosmology(struct cosmology *cosmo, const char *fname) {
    cosmo->h = ini_getd("Cosmology", "h", 0.70, fname);
    cosmo->n_s = ini_getd("Cosmology", "n_s", 0.97, fname);
    cosmo->A_s = ini_getd("Cosmology", "A_s", 2.215e-9, fname);
    cosmo->k_pivot = ini_getd("Cosmology", "k_pivot", 0.05, fname);

    return 0;
}

double primordialPower(const double k, const struct cosmology *cosmo) {
    if (k == 0) return 0;

    double A_s = cosmo->A_s;
    double n_s = cosmo->n_s;
    double k_pivot = cosmo->k_pivot;

    return A_s * pow(k/k_pivot, n_s - 1.) * k * (2. * M_PI * M_PI);
}
