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
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/params.h"

/* The .ini parser library is minIni */
#include "../parser/minIni.h"

int readParams(struct params *pars, const char *fname) {
     pars->Seed = ini_getl("Random", "Seed", 1, fname);
     pars->GridSize = ini_getl("Simulation", "GridSize", 64, fname);
     pars->BoxLength = ini_getd("Simulation", "BoxLength", 1000, fname);
     pars->z_start = ini_getd("Simulation", "z_start", 1000, fname);
    
     /* Read strings */
     int len = DEFAULT_STRING_LENGTH;
     pars->TransferFunctionsFile = malloc(len);
     ini_gets("TransferFunctions", "File", "", pars->TransferFunctionsFile, len, fname);
    
     return 0;
}


int cleanParams(struct params *pars) {
    free(pars->TransferFunctionsFile);
    
    return 0;
}
