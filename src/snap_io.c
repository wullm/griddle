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

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/snap_io.h"

/* The current limit of for parallel HDF5 writes is 2GB */
#define HDF5_PARALLEL_LIMIT 2147000000LL
/* The default chunk size, corresponding to ~0.5 MB */
#define HDF5_CHUNK_SIZE 65536LL

int exportSnapshot(struct params *pars, struct units *us,
                   struct particle *particles, int output_num, double a,
                   int N, long long int local_partnum) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Particle counts */
    long long int total_partnum = N * N * N;
    long long int parts_in_file = total_partnum;

    /* The ExportName */
    const char *ExportName = "PartType1"; // cdm

    /* Vector dataspace (e.g. positions, velocities) */
    const hsize_t vrank = 2;
    const hsize_t vdims[2] = {parts_in_file, 3};
    hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);

    /* Scalar dataspace (e.g. masses, particle ids) */
    const hsize_t srank = 1;
    const hsize_t sdims[1] = {parts_in_file};
    hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

    /* Set chunking for vectors */
    hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t vchunk[2] = {HDF5_CHUNK_SIZE, 3};
    H5Pset_chunk(h_prop_vec, vrank, vchunk);

    /* Set chunking for scalars */
    hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t schunk[1] = {HDF5_CHUNK_SIZE};
    H5Pset_chunk(h_prop_sca, srank, schunk);

    /* Form the filename */
    char fname[DEFAULT_STRING_LENGTH];
    sprintf(fname, "%s_%04d.hdf5", pars->SnapshotBaseName, output_num);

    if (rank == 0) {
        /* Create the output file */
        hid_t h_out_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        /* Writing attributes into the Header & Cosmology groups */
        int err = writeHeaderAttributes(pars, us, a, parts_in_file, total_partnum, h_out_file);
        if (err > 0) exit(1);

        /* The particle group in the output file */
        hid_t h_grp;

        /* Datsets */
        hid_t h_data;

        /* Create the particle group in the output file */
        h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Coordinates (use vector space) */
        h_data = H5Dcreate(h_grp, "Coordinates", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Velocities (use vector space) */
        h_data = H5Dcreate(h_grp, "Velocities", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Particle IDs (use scalar space) */
        h_data = H5Dcreate(h_grp, "ParticleIDs", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Masses (use scalar space) */
        h_data = H5Dcreate(h_grp, "Masses", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Close the group */
        H5Gclose(h_grp);

        /* Close the file */
        H5Fclose(h_out_file);
    }

    /* Wait until all ranks are finished */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, MPI_COMM_WORLD, MPI_INFO_NULL);

    /* Open the hdf5 file */
    hid_t h_out_file = H5Fopen(fname, H5F_ACC_RDWR, prop_faxs);
    H5Pclose(prop_faxs);

    /* The particle group in the output file */
    hid_t h_grp = H5Gopen(h_out_file, ExportName, H5P_DEFAULT);

    /* Create vector & scalar datapsace for smaller chunks of data */
    const hsize_t ch_vdims[2] = {local_partnum, 3};
    const hsize_t ch_sdims[1] = {local_partnum};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* Determine the number of particles on each rank */
    long long int *partnum_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    long long int *first_id_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    partnum_by_rank[rank] = local_partnum;
    MPI_Allreduce(MPI_IN_PLACE, partnum_by_rank, MPI_Rank_Count, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    /* Determine the start of the hyperslab corresponding to each rank */
    for (int i = 1; i < MPI_Rank_Count; i++) {
        first_id_by_rank[i] = first_id_by_rank[i - 1] + partnum_by_rank[i - 1];
    }

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = first_id_by_rank[rank];
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};

    /* Choose the corresponding hyperslabs inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Unpack the remaining particle data into contiguous arrays */
    double *coords = malloc(3 * local_partnum * sizeof(double));
    double *vels = malloc(3 * local_partnum * sizeof(double));
    long long *ids = malloc(1 * local_partnum * sizeof(long long));
    double *masses = malloc(1 * local_partnum * sizeof(double));
    for (long long i = 0; i < local_partnum; i++) {
        coords[i * 3 + 0] = particles[i].x[0];
        coords[i * 3 + 1] = particles[i].x[1];
        coords[i * 3 + 2] = particles[i].x[2];
        vels[i * 3 + 0] = particles[i].v[0];
        vels[i * 3 + 1] = particles[i].v[1];
        vels[i * 3 + 2] = particles[i].v[2];
        ids[i] = particles[i].id;
        masses[i] = particles[i].m;
    }

    /* Write coordinate data (vector) */
    hid_t h_data = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, coords);
    H5Dclose(h_data);
    free(coords);

    /* Write velocity data (vector) */
    h_data = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, vels);
    H5Dclose(h_data);
    free(vels);

    /* Write particle id data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleIDs", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, H5P_DEFAULT, ids);
    H5Dclose(h_data);
    free(ids);

    /* Write mass data (scalar) */
    h_data = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, masses);
    H5Dclose(h_data);
    free(masses);

    /* Close the group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_out_file);

    return 0;
}

int writeHeaderAttributes(struct params *pars, struct units *us, double a,
                          long long int Npart_local, long long int Npart_total,
                          hid_t h_file) {

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Retrieve the number of ranks */
    int MPI_Rank_Count;
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims_three[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims_three, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxlen = pars->BoxLength;
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);
    H5Aclose(h_attr);

    /* Change dataspace dimensions to scalar value attributes */
    const hsize_t adims_single[1] = {1};
    H5Sset_extent_simple(h_aspace, arank, adims_single, NULL);

    /* Create the Dimension attribute and write the data */
    int dimension = 3;
    h_attr = H5Acreate1(h_grp, "Dimension", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &dimension);
    H5Aclose(h_attr);

    /* Create the Redshift attribute and write the data */
    double z_output = 1./a - 1;
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &z_output);
    H5Aclose(h_attr);

    /* Create the Flag_Entropy_ICs attribute and write the data */
    int flag_entropy = 0;
    h_attr = H5Acreate1(h_grp, "Flag_Entropy_ICs", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &flag_entropy);
    H5Aclose(h_attr);

    /* Create the NumFilesPerSnapshot attribute and write the data */
    int num_files_per_snapshot = 1;
    h_attr = H5Acreate1(h_grp, "NumFilesPerSnapshot", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &num_files_per_snapshot);
    H5Aclose(h_attr);

    /* Change dataspace dimensions to particle type attributes */
    const hsize_t adims_pt[1] = {7}; //particle type 0-6
    H5Sset_extent_simple(h_aspace, arank, adims_pt, NULL);

    /* Collect particle type attributes using the ExportNames */
    long long int numparts_local[7] = {0, 0, 0, 0, 0, 0, Npart_local};
    long long int numparts_total[7] = {0, 0, 0, 0, 0, 0, Npart_total};
    long long int numparts_high_word[7] = {0, 0, 0, 0, 0, 0, Npart_total >> 32};
    double mass_table[7] = {0., 0., 0., 0., 0., 0., 0.};

    /* Create the NumPart_ThisFile attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_ThisFile", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_local);
    H5Aclose(h_attr);

    /* Create the NumPart_Total attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_Total", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_total);
    H5Aclose(h_attr);

    /* Create the NumPart_Total_HighWord attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_Total_HighWord", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_high_word);
    H5Aclose(h_attr);

    /* Create the MassTable attribute and write the data */
    h_attr = H5Acreate1(h_grp, "MassTable", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, mass_table);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Header group */
    H5Gclose(h_grp);

    /* Create the Cosmology group */
    h_grp = H5Gcreate(h_file, "/Cosmology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Create the Redshift attribute and write the data */
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &z_output);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);

    /* Create the Units group */
    h_grp = H5Gcreate(h_file, "/Units", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Determine the units used */
    double unit_mass_cgs = us->UnitMassKilogram * 1000;
    double unit_length_cgs = us->UnitLengthMetres * 100;
    double unit_time_cgs = us->UnitTimeSeconds;
    double unit_temperature_cgs = us->UnitTemperatureKelvin;
    double unit_current_cgs = us->UnitCurrentAmpere;

    /* Write the internal unit system */
    h_attr = H5Acreate1(h_grp, "Unit mass in cgs (U_M)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_mass_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit length in cgs (U_L)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_length_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit time in cgs (U_t)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_time_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit temperature in cgs (U_T)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_temperature_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit current in cgs (U_I)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_current_cgs);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);


    return 0;
}

int readSnapshot(struct params *pars, struct units *us,
                 struct particle *particles, const char *fname,
                 long long int *local_partnum, long long int max_partnum) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, MPI_COMM_WORLD, MPI_INFO_NULL);

    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, prop_faxs);
    H5Pclose(prop_faxs);

    /* The ExportName */
    const char *ExportName = "PartType1"; // cdm

    /* Open the particle group */
    hid_t h_grp = H5Gopen(h_file, ExportName, H5P_DEFAULT);

    /* Open the coordinates dataset */
    hid_t h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

    /* Find the dataspace (in the file) */
    hid_t h_space = H5Dget_space (h_dat);

    /* Get the dimensions of this dataspace */
    hsize_t dims[2];
    H5Sget_simple_extent_dims(h_space, dims, NULL);

    /* The total number of particles present in the file */
    hid_t N_tot = dims[0];

    /* The number of particles that will be read per rank */
    hid_t N_per_rank = (long int) (N_tot / MPI_Rank_Count);
    hid_t N_this_rank = (rank < MPI_Rank_Count - 1) ? N_per_rank : N_tot - (MPI_Rank_Count - 1) * N_per_rank;
    *local_partnum = N_this_rank;

    /* The address of the first particle to be read on this rank */
    hid_t localFirstNumber = rank * N_per_rank;

    // printf("%d: we will read %ld particles, starting from %ld\n", rank, N_this_rank, localFirstNumber);

    /* Close the data and memory spaces */
    H5Sclose(h_space);

    /* Close the dataset */
    H5Dclose(h_dat);

    /* Define the hyperslab */
    hsize_t slab_dims[2], start[2]; //for 3-vectors
    hsize_t slab_dims_one[1], start_one[1]; //for scalars

    /* Slab dimensions for 3-vectors */
    slab_dims[0] = N_this_rank;
    slab_dims[1] = 3; //(x,y,z)
    start[0] = localFirstNumber;
    start[1] = 0; //start with x

    /* Slab dimensions for scalars */
    slab_dims_one[0] = N_this_rank;
    start_one[0] = localFirstNumber;

    /* Open the coordinates dataset and corresponding data space */
    h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
    h_space = H5Dget_space (h_dat);

    /* Select the hyperslab */
    hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                       NULL, slab_dims, NULL);

    /* Create a memory space */
    hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

    /* Create the data array and read the data */
    double *coord_data = malloc(N_this_rank * 3 * sizeof(double));
    status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                     coord_data);

     /* Close the memory space, data space, data set */
     H5Sclose(h_mems);
     H5Sclose(h_space);
     H5Dclose(h_dat);

    /* Transfer the coordinate array to the particles */
    for (int i = 0; i < *local_partnum; i++) {
        particles[i].x[0] = coord_data[i * 3 + 0];
        particles[i].x[1] = coord_data[i * 3 + 1];
        particles[i].x[2] = coord_data[i * 3 + 2];
    }

    free(coord_data);

    /* Open the masses dataset and corresponding data space */
    h_dat = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
    h_space = H5Dget_space (h_dat);

    /* Select the hyperslab */
    status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                        slab_dims_one, NULL);

    /* Create a memory space */
    h_mems = H5Screate_simple(1, slab_dims_one, NULL);

    /* Create the data array and read the data */
    double *mass_data = malloc(N_this_rank * sizeof(double));
    status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                     mass_data);

    /* Close the memory space, data space, data set */
    H5Sclose(h_mems);
    H5Sclose(h_space);
    H5Dclose(h_dat);

    /* Transfer the contiguous array to the particle data */
    for (int i = 0; i < *local_partnum; i++) {
        particles[i].m = mass_data[i];
    }

    /* Free the contiguous array */
    free(mass_data);

    /* Open the velocities dataset and corresponding dataspace */
    h_dat = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
    h_space = H5Dget_space(h_dat);

    /* Select the hyperslab */
    status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                 NULL, slab_dims, NULL);

    /* Create a memory space */
    h_mems = H5Screate_simple(2, slab_dims, NULL);

    /* Create the data array and read the data */
    double *veloc_data = malloc(N_this_rank * 3 * sizeof(double));
    status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT, veloc_data);

    /* Close the memory space, data space, data set */
    H5Sclose(h_mems);
    H5Sclose(h_space);
    H5Dclose(h_dat);

    /* Transfer the contiguous array to the particle data */
    for (int i = 0; i < *local_partnum; i++) {
        particles[i].v[0] = veloc_data[i * 3 + 0];
        particles[i].v[1] = veloc_data[i * 3 + 1];
        particles[i].v[2] = veloc_data[i * 3 + 2];
    }

    /* Free the contiguous array */
    free(veloc_data);

    /* Close the particle group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    assert(status >= 0);

    return 0;

}
