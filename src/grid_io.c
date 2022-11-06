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

/* Methods for input and output of data cubes, using the HDF5 format */

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/grid_io.h"

#define HDF5_PARALLEL_IO_MAX_BYTES 2000000000LL

/* Read a data cube from disk, allocating memory as we go */
int readFieldFile(double **box, int *N, double *box_len, const char *fname) {
    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the size of the field */
    hid_t h_attr, h_err;
    double boxsize[3];

    /* Open and read out the attribute */
    h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &boxsize);
    if (h_err < 0) {
        printf("Error reading hdf5 attribute '%s'.\n", "BoxSize");
        return 1;
    }

    /* It should be a cube */
    assert(boxsize[0] == boxsize[1]);
    assert(boxsize[1] == boxsize[2]);
    *box_len = boxsize[0];

    /* Close the attribute, and the Header group */
    H5Aclose(h_attr);
    H5Gclose(h_grp);

    /* Open the Field group */
    h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Open the dataspace and fetch the grid dimensions */
    hid_t h_space = H5Dget_space(h_data);
    int ndims = H5Sget_simple_extent_ndims(h_space);
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);
    int read_N = dims[0];

    /* We should be in 3D */
    if (ndims != 3) {
        printf("Number of dimensions %d != 3.\n", ndims);
        return 2;
    }
    /* It should be a cube (but allow for padding in the last dimension) */
    if (read_N != dims[1] || (read_N != dims[2] && (read_N+2) != dims[2])) {
        printf("Non-cubic grid size (%lld, %lld, %lld).\n", dims[0], dims[1], dims[2]);
        return 2;
    }
    /* Store the grid size */
    *N = read_N;

    /* Allocate the array (without padding) */
    *box = malloc(read_N * read_N * read_N * sizeof(double));

    /* The hyperslab that should be read (needed in case of padding) */
    const hsize_t space_rank = 3;
    const hsize_t space_dims[3] = {read_N, read_N, read_N}; //3D space

    /* Offset of the hyperslab */
    const hsize_t space_offset[3] = {0, 0, 0};

    /* Create memory space for the chunk */
    hid_t h_memspace = H5Screate_simple(space_rank, space_dims, NULL);
    H5Sselect_hyperslab(h_space, H5S_SELECT_SET, space_offset, NULL, space_dims, NULL);

    /* Read out the data */
    h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, *box);
    if (h_err < 0) {
        printf("Error reading hdf5 file '%s'.\n", fname);
        return 1;
    }

    /* Close the dataspaces and dataset */
    H5Sclose(h_memspace);
    H5Sclose(h_space);
    H5Dclose(h_data);

    /* Close the Field group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    /* Free memory */
    free(dims);

    return 0;
}

/* Write a data cube to disk in HDF5 format without any compression */
int writeFieldFile(const double *box, int N, double boxlen, const char *fname) {
    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);

    /* Close the attribute, corresponding dataspace, and the Header group */
    H5Aclose(h_attr);
    H5Sclose(h_aspace);
    H5Gclose(h_grp);

    /* Create the Field group */
    h_grp = H5Gcreate(h_file, "/Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for the field */
    const hsize_t frank = 3;
    const hsize_t fdims[3] = {N, N, N}; //3D space
    hid_t h_fspace = H5Screate_simple(frank, fdims, NULL);

    /* Create the dataset for the field */
    hid_t h_data = H5Dcreate(h_grp, "Field", H5T_NATIVE_DOUBLE, h_fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Write the data */
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_fspace, h_fspace, H5P_DEFAULT, box);

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_fspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}

/* Write a field file as floats with a lossy compression filter, specified
 * by the number of digits after the decimal to keep. (The rest will be filled
 * with rubbish). */
int writeFieldFileCompressed(const double *box, int N, double boxlen,
                             const char *fname, int digits) {
    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);

    /* Close the attribute, corresponding dataspace, and the Header group */
    H5Aclose(h_attr);
    H5Sclose(h_aspace);
    H5Gclose(h_grp);

    /* Create the Field group */
    h_grp = H5Gcreate(h_file, "/Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for the field */
    const hsize_t frank = 3;
    const hsize_t fdims[3] = {N, N, N}; //3D space
    hid_t h_fspace = H5Screate_simple(frank, fdims, NULL);

    /* Determine chunk size (chosen to get ~1MB per chunk) */
    hsize_t chunk_size[3];
    if (N < 64) {
        chunk_size[0] = chunk_size[1] = chunk_size[2] = N;
    } else {
        chunk_size[0] = chunk_size[1] = chunk_size[2] = 64;
    }

    /* Prepare the dataset properties */
    hid_t h_prop = H5Pcreate(H5P_DATASET_CREATE);

    /* Set chunking */
    H5Pset_chunk(h_prop, 3, chunk_size);

    /* Set lossy filter (keep "digits" digits after the decimal point) */
    if (digits > 0) {
        H5Pset_scaleoffset(h_prop, H5Z_SO_FLOAT_DSCALE, digits);
    }

    /* Set shuffle and lossless compression filters (GZIP level 4) */
    H5Pset_shuffle(h_prop);
    H5Pset_deflate(h_prop, 4);

    /* Create the dataset for the field */
    hid_t h_data = H5Dcreate(h_grp, "Field", H5T_NATIVE_FLOAT, h_fspace,
                             H5P_DEFAULT, h_prop, H5P_DEFAULT);

    /* Write the data */
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_fspace, h_fspace, H5P_DEFAULT, box);

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_fspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}

/* Write a distributed data cube to disk in HDF5 format without any compression */
int writeFieldFile_dg(struct distributed_grid *dg, const char *fname) {
    if (dg->momentum_space == 1) {
        printf("Error: attempting to export while in momentum space.\n");
        return 1;
    }

    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, MPI_COMM_WORLD, MPI_INFO_NULL);

    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, prop_faxs);
    H5Pclose(prop_faxs);

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {dg->boxlen, dg->boxlen, dg->boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);

    /* Close the attribute, corresponding dataspace, and the Header group */
    H5Aclose(h_attr);
    H5Sclose(h_aspace);
    H5Gclose(h_grp);

    if (dg->NX * dg->N * (dg->N+2) * sizeof(double) > HDF5_PARALLEL_IO_MAX_BYTES) {
        printf("Error: parallel HDF5 cannot handle more than 2GB per chunk.\n");
        return 1;
    }

    /* Create the Field group */
    h_grp = H5Gcreate(h_file, "/Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for the field */
    const hsize_t frank = 3;
    const hsize_t fdims[3] = {dg->N, dg->N, dg->N+2}; //3D space
    hid_t h_fspace = H5Screate_simple(frank, fdims, NULL);

    /* Create the dataset for the field */
    hid_t h_data = H5Dcreate(h_grp, "Field", H5T_GRID_TYPE, h_fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* The chunk in question */
    const hsize_t chunk_rank = 3;
    const hsize_t chunk_dims[3] = {dg->NX, dg->N, dg->N+2}; //3D space

    /* Offset of the chunk inside the grid */
    const hsize_t chunk_offset[3] = {dg->X0, 0, 0};

    /* Create memory space for the chunk */
    hid_t h_memspace = H5Screate_simple(chunk_rank, chunk_dims, NULL);
    H5Sselect_hyperslab(h_fspace, H5S_SELECT_SET, chunk_offset, NULL, chunk_dims, NULL);

    /* Write the data */
    hid_t h_err = H5Dwrite(h_data, H5T_GRID_TYPE, h_memspace, h_fspace, H5P_DEFAULT, dg->box);
    if (h_err < 0) {
        printf("Error: writing chunk of hdf5 data.\n");
        return 1;
    }

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_fspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}
