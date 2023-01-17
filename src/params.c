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

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/params.h"

/* The .ini parser library is minIni */
#include "../parser/minIni.h"


/* Here used_parameter_fname is the name of the new "used parameters file" that
 * is being created, while fname is the filename of the input parameter file. */
void begin_used_parameter_file(const char *used_parameter_fname,
                               const char *fname, FILE **used_parameters_f,
                               int rank) {
    if (rank == 0) {
        /* Get the current time */
        char timestring[26];
        struct timeval tv;
        gettimeofday(&tv, NULL);
        struct tm* tm_info = localtime(&tv.tv_sec);
        strftime(timestring, 26, "%Y-%m-%d %H:%M:%S (%Z)", tm_info);

        /* Create a formatted file with the used parameters */
        *used_parameters_f = fopen(used_parameter_fname, "w");
        fprintf(*used_parameters_f, "# Parameters read from '%s'\n", fname);
        fprintf(*used_parameters_f, "# Parameters read at %s\n", timestring);
    }
}

void begin_section(const char *section, FILE *used_parameters_f, int rank) {
    /* Print to a formatted parameter file */
    if (rank == 0) {
        fprintf(used_parameters_f, "\n[%s]\n", section);
    }
}

long read_long(const char *section, const char *key, long default_value,
               const char *fname, FILE *used_parameters_f, int rank) {

    /* Read the parameter */
    long value = ini_getl(section, key, default_value, fname);

    /* Print to a formatted parameter file */
    if (rank == 0) {
        char parstring[DEFAULT_STRING_LENGTH];
        sprintf(parstring, "%s = %ld", key, value);
        if (value == default_value) {
            fprintf(used_parameters_f, "%-50s # (default)\n", parstring);
        } else {
            fprintf(used_parameters_f, "%s\n", parstring);
        }
    }

    return value;
}

double read_double(const char *section, const char *key, double default_value,
                   const char *fname, FILE *used_parameters_f, int rank) {

    /* Read the parameter */
    double value = ini_getd(section, key, default_value, fname);

    /* Print to a formatted parameter file */
    if (rank == 0) {
        char parstring[DEFAULT_STRING_LENGTH];
        sprintf(parstring, "%s = %g", key, value);
        if (value == default_value) {
            fprintf(used_parameters_f, "%-50s # (default)\n", parstring);
        } else {
            fprintf(used_parameters_f, "%s\n", parstring);
        }
    }

    return value;
}

int read_string(const char *section, const char *key, const char *default_value,
                char *buffer, int buffer_size, const char *fname,
                FILE *used_parameters_f, int rank) {

    /* Read the parameter */
    ini_gets(section, key, default_value, buffer, buffer_size, fname);

    /* Print to a formatted parameter file */
    if (rank == 0) {
        char parstring[DEFAULT_STRING_LENGTH];
        sprintf(parstring, "%s = %s", key, buffer);
        if (strcmp(buffer, default_value) == 0) {
            fprintf(used_parameters_f, "%-50s # (default)\n", parstring);
        } else {
            fprintf(used_parameters_f, "%s\n", parstring);
        }
    }

    return 0;
}

int readParams(struct params *pars, const char *fname) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Create a formatted file with the used parameters */
    FILE *f = NULL;
    begin_used_parameter_file("used_parameters.ini", fname, &f, rank);

    /* Prepare string parameters */
    int len = DEFAULT_STRING_LENGTH;
    pars->InitialConditionsFile = malloc(len);
    pars->TransferFunctionsFile = malloc(len);
    pars->SnapshotTimesString = malloc(len);
    pars->SnapshotBaseName = malloc(len);
    pars->PowerSpectrumTimesString = malloc(len);
    pars->PowerSpectrumTypes = malloc(len);
    pars->HaloFindingTimesString = malloc(len);
    pars->CatalogueBaseName = malloc(len);
    pars->SnipBaseName = malloc(len);

    /* Random number generator parameters */
    begin_section("Random", f, rank);
    pars->Seed = read_long("Random", "Seed", 1, fname, f, rank);
    pars->FixedModes = read_long("Random", "FixedModes", 0, fname, f, rank);
    pars->InvertedModes = read_long("Random", "InvertedModes", 0, fname, f, rank);

    /* Initial conditions parameters */
    begin_section("InitialConditions", f, rank);
    pars->GenerateICs = read_long("InitialConditions", "Generate", 1, fname, f, rank);
    pars->DoNewtonianBackscaling = read_long("InitialConditions", "DoNewtonianBackscaling", 1, fname, f, rank);
    read_string("InitialConditions", "File", "", pars->InitialConditionsFile, len, fname, f, rank);

    /* Transfer funtions parameters */
    begin_section("TransferFunctions", f, rank);
    read_string("TransferFunctions", "File", "", pars->TransferFunctionsFile, len, fname, f, rank);

    /* Simulation parameters */
    begin_section("Simulation", f, rank);
    pars->PartGridSize = read_long("Simulation", "PartGridSize", 64, fname, f, rank);
    pars->MeshGridSize = read_long("Simulation", "MeshGridSize", 64, fname, f, rank);
    pars->NeutrinosPerDim = read_long("Simulation", "NeutrinosPerDim", 0, fname, f, rank);
    pars->ForeignBufferSize = read_long("Simulation", "ForeignBufferSize", 3000000, fname, f, rank);
    pars->BoxLength = read_double("Simulation", "BoxLength", 1000.0, fname, f, rank);
    pars->ScaleFactorBegin = read_double("Simulation", "ScaleFactorBegin", 0.03125, fname, f, rank);
    pars->ScaleFactorEnd = read_double("Simulation", "ScaleFactorEnd", 1.0, fname, f, rank);
    pars->ScaleFactorStep = read_double("Simulation", "ScaleFactorStep", 0.05, fname, f, rank);
    pars->ScaleFactorTarget = read_double("Simulation", "ScaleFactorTarget", 1.0, fname, f, rank);
    pars->DerivativeOrder = read_long("Simulation", "DerivativeOrder", 4, fname, f, rank);

    /* Neutrino parameters */
    begin_section("Neutrino", f, rank);
    pars->NeutrinoPreIntegration = read_long("Neutrino", "PreIntegration", 1, fname, f, rank);
    pars->NeutrinoScaleFactorEarly = read_double("Neutrino", "ScaleFactorEarly", 0.005, fname, f, rank);
    pars->NeutrinoScaleFactorEarlyStep = read_double("Neutrino", "ScaleFactorEarlyStep", 0.2, fname, f, rank);

    /* Snapshot parameters */
    begin_section("Snapshots", f, rank);
    read_string("Snapshots", "OutputTimes", "", pars->SnapshotTimesString, len, fname, f, rank);
    read_string("Snapshots", "BaseName", "snap", pars->SnapshotBaseName, len, fname, f, rank);

    /* Power spectrum parameters */
    begin_section("PowerSpectra", f, rank);
    read_string("PowerSpectra", "OutputTimes", "", pars->PowerSpectrumTimesString, len, fname, f, rank);
    read_string("PowerSpectra", "Types", "all", pars->PowerSpectrumTypes, len, fname, f, rank);
    pars->PowerSpectrumBins = read_long("PowerSpectra", "PowerSpectrumBins", 50, fname, f, rank);
    pars->PositionDependentSplits = read_long("PowerSpectra", "PositionDependentSplits", 8, fname, f, rank);

    /* Halo finding parameters */
    begin_section("HaloFinding", f, rank);
    read_string("HaloFinding", "OutputTimes", "", pars->HaloFindingTimesString, len, fname, f, rank);
    read_string("HaloFinding", "BaseName", "catalogue", pars->CatalogueBaseName, len, fname, f, rank);
    pars->LinkingLength = read_double("HaloFinding", "LinkingLength", 0.2, fname, f, rank);
    pars->MinHaloParticleNum = read_long("HaloFinding", "MinHaloParticleNum", 20, fname, f, rank);
    pars->DoSphericalOverdensities = read_long("HaloFinding", "DoSphericalOverdensities", 1, fname, f, rank);
    pars->FOFBufferSize =  read_long("HaloFinding", "FOFBufferSize", 10000, fname, f, rank);
    pars->HaloFindCellNumber = read_long("HaloFinding", "CellNumber", 256, fname, f, rank);
    pars->SphericalOverdensityThreshold = read_double("HaloFinding", "SphericalOverdensityThreshold", 200.0, fname, f, rank);
    pars->SphericalOverdensityMinLookRadius = read_double("HaloFinding", "SphericalOverdensityMinLookRadius", 10.0, fname, f, rank);
    pars->ShrinkingSphereInitialRadius = read_double("HaloFinding", "ShrinkingSphereInitialRadius", 0.9, fname, f, rank);
    pars->ShrinkingSphereRadiusFactorCoarse = read_double("HaloFinding", "ShrinkingSphereRadiusFactorCoarse", 0.75, fname, f, rank);
    pars->ShrinkingSphereRadiusFactor = read_double("HaloFinding", "ShrinkingSphereRadiusFactor", 0.95, fname, f, rank);
    pars->ShrinkingSphereMassFraction = read_double("HaloFinding", "ShrinkingSphereMassFraction", 0.01, fname, f, rank);
    pars->ShrinkingSphereMinParticleNum = read_long("HaloFinding", "ShrinkingSphereMinParticleNum", 100, fname, f, rank);
    pars->ExportSnipshots = read_long("HaloFinding", "ExportSnipshots", 1, fname, f, rank);
    pars->SnipshotReduceFactor =  read_double("HaloFinding", "SnipshotReduceFactor", 0.01, fname, f, rank);
    pars->SnipshotMinParticleNum =  read_long("HaloFinding", "SnipshotMinParticleNum", 5, fname, f, rank);
    pars->SnipshotPositionDScaleCompression =  read_long("HaloFinding", "SnipshotPositionDScaleCompression", 3, fname, f, rank);
    pars->SnipshotVelocityDScaleCompression =  read_long("HaloFinding", "SnipshotVelocityDScaleCompression", 1, fname, f, rank);
    pars->SnipshotZipCompressionLevel =  read_long("HaloFinding", "SnipshotZipCompressionLevel", 4, fname, f, rank);
    read_string("HaloFinding", "SnipBaseName", "snip", pars->SnipBaseName, len, fname, f, rank);

    /* Close the formatted parameter file */
    if (rank == 0) {
        fclose(f);
    }

    return 0;
}

int cleanParams(struct params *pars) {
    free(pars->InitialConditionsFile);
    free(pars->TransferFunctionsFile);
    free(pars->SnapshotTimesString);
    free(pars->SnapshotBaseName);
    free(pars->PowerSpectrumTimesString);
    free(pars->PowerSpectrumTypes);
    free(pars->HaloFindingTimesString);
    free(pars->CatalogueBaseName);
    free(pars->SnipBaseName);

    return 0;
}

int parseArrayString(char *string, double **array, int *length) {
    /* Check that there is anything there */
    if (strlen(string) <= 0) {
        *length = 0;
        *array = NULL;
        return 0;
    }

    /* Permissible delimiters */
    char delimiters[] = " ,\t\n";

    /* Make a copy of the string before it gets modified */
    char copy[DEFAULT_STRING_LENGTH];
    sprintf(copy, "%s", string);

    /* Count the number of values */
    char *token = strtok(string, delimiters);
    int count = 0;
    while (token != NULL) {
        count++;
        token = strtok(NULL, delimiters);
    }

    *length = count;

    if (count == 0)
        return 0;

    /* Allocate memory and return the length */
    *array = calloc(count, sizeof(double));

    /* Parse the original string again */
    token = strtok(copy, delimiters);
    for (int i = 0; i < count; i++) {
        sscanf(token, "%lf", &(*array)[i]);
        token = strtok(NULL, delimiters);
    }

    return 0;
}

int parseOutputList(char *string, double **output_list, int *num_outputs,
                    double a_begin, double a_end) {

    /* Parse the output times from the input string */
    parseArrayString(string, output_list, num_outputs);

    if (*num_outputs < 1) {
        return 0;
    } else if (*num_outputs > 1) {
        /* Check that the output times are in ascending order */
        for (int i = 1; i < *num_outputs; i++) {
            if ((*output_list)[i] <= (*output_list)[i - 1]) {
                printf("Output times should be in strictly ascending order.\n");
                exit(1);
            }
        }
    }

    /* Check that the first output is after the beginning and the last before the end */
    if ((*output_list)[0] < a_begin) {
        printf("The first output should be after the start of the simulation.\n");
        exit(1);
    } else if ((*output_list)[*num_outputs - 1] > a_end) {
        printf("The last output should be before the end of the simulation.\n");
        exit(1);
    }

    return 0;
}

int parseCharArrayString(char *string, char ***array, int *length) {
    /* Check that there is anything there */
    if (strlen(string) <= 0) {
        *length = 0;
        *array = NULL;
        return 0;
    }

    /* Permissible delimiters */
    char delimiters[] = " ,\t\n";

    /* Make a copy of the string before it gets modified */
    char copy[DEFAULT_STRING_LENGTH];
    sprintf(copy, "%s", string);

    /* Count the number of values */
    char *token = strtok(string, delimiters);
    int count = 0;
    while (token != NULL) {
        count++;
        token = strtok(NULL, delimiters);
    }

    *length = count;

    if (count == 0)
        return 0;

    /* Allocate memory and return the length */
    *array = calloc(count, sizeof(char*));
    int offset = 0;

    /* Parse the original string again */
    token = strtok(copy, delimiters);
    for (int i = 0; i < count; i++) {
        char value[DEFAULT_STRING_LENGTH];
        int len;

        /* Parse the sub-string */
        sscanf(token, "%s", value);
        token = strtok(NULL, delimiters);
        len = strlen(value);

        /* Insert it back into the input string and terminate it with null */
        strcpy(string + offset, value);
        string[offset + len + 1] = '\0';

        /* Store and then increment the pointer */
        (*array)[i] = string + offset;
        offset += len + 1;
    }

    return 0;
}

int parseGridTypeList(char *string, enum grid_type **type_list, int *num_types) {
    /* Parse the grid types */
    char **types_strings = NULL;
    parseCharArrayString(string, &types_strings, num_types);

    if (*num_types < 1)
        return 0;

    /* Parse the requested grid types */
    *type_list = malloc(*num_types * sizeof(enum grid_type));
    for (int i = 0; i < *num_types; i++) {
        /* Compare with possible grid types */
        int match_found = 0;
        for (int j = 0; j < num_grid_types; j++) {
            if (strcmp(types_strings[i], grid_type_names[j]) == 0) {
                (*type_list)[i] = j;
                match_found = 1;
            }
        }

        if (!match_found) {
            printf("Error: unknown power spectrum type: %s!\n", types_strings[i]);
            exit(1);
        }
    }
    free(types_strings);

    return 0;
}