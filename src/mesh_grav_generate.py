# Automatically generate code for fast differentiation of grids using
# central difference and CIC interpolation

import numpy as np

minval = 1

# Determine all the cells that are needed for CIC
i = np.arange(2)
X, Y, Z = np.meshgrid(i, i, i)
X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

def count_cic(indices, positions, coefficients, x, y, z, coeff):
    for i, j, k in zip(X, Y, Z):
            indices.append([x + i, y + j, z + k])
            positions.append([i, j, k])
            coefficients.append(coeff)

# Loop over the finite difference stencil along each dimension,
# keeping track of the cells that are to be evaluated
def count_fd_stencil(index, coeff):
    indices_x = []
    indices_y = []
    indices_z = []
    positions_x = []
    positions_y = []
    positions_z = []
    coefficients_x = []
    coefficients_y = []
    coefficients_z = []
    for i, c in zip(index, coeff):
        count_cic(indices_x, positions_x, coefficients_x, i, 0, 0, c)
        count_cic(indices_y, positions_y, coefficients_y, 0, i, 0, c)
        count_cic(indices_z, positions_z, coefficients_z, 0, 0, i, c)
    return (indices_x, indices_y, indices_z, positions_x, positions_y,
            positions_z, coefficients_x, coefficients_y, coefficients_z)

# Generate the variable name for the cell value at position (i, j, k)
def generate_cell_name(index):
    return ("val_" + "".join([str(j+minval) for j in index]))

# Generate the code for accumulating the central difference values
def generate_accumulator(var_name, indices, positions, coefficients):
    for i, (ind,pos,c) in enumerate(zip(indices, positions, coefficients)):
        cell_name = generate_cell_name(ind)
        fraction_str = "".join([("d" if j else "t") for j in pos])
        if c == -1:
            print(var_name + " -= " + cell_name + " * " + fraction_str + ";")
        elif c == 1:
            print(var_name + " += " + cell_name + " * " + fraction_str + ";")
        elif c < 0:
            print(var_name + " -= " + cell_name + " * " + fraction_str + " * " + str(-c) + ";")
        elif c > 0:
            print(var_name + " += " + cell_name + " * " + fraction_str + " * " + str(c) + ";")

# The product of the fractional displacements from the grid centre\
print("/* Products of fractional displacements from cell corners */")
dims = ["x", "y", "z"]
for i, j, k in zip(X, Y, Z):
        pos = [i, j, k]
        var_name = "".join(["d" if u else "t" for u in pos])
        result = " * ".join([("d" if u else "t") + dim for (u, dim) in zip(pos, dims)])
        print("double " + var_name + " = " + result + ";")

# First-order accurate forward difference
index_fio = [1, 0]
coeff_fio = [1.0, -1.0]

# Second-order accurate differentation (modulo a factor 2)
index_so = [1, -1]
coeff_so = [1.0, -1.0]

# Fourth-order accurate differentation (modulo a factor 12)
index_fo = [2, 1, -1, -2]
coeff_fo = [-1.0, 8.0, -8.0, 1.0]

if (True):
    # Generate the second-order accutate functions
    (ind_x, ind_y, ind_z,
    pos_x, pos_y, pos_z,
    coeff_x, coeff_y, coeff_z) = count_fd_stencil(index_so, coeff_so)
else:
    # Generate the fourth-order accutate functions
    (ind_x, ind_y, ind_z,
    pos_x, pos_y, pos_z,
    coeff_x, coeff_y, coeff_z) = count_fd_stencil(index_fo, coeff_fo)

# Find all the cells that are needed accross the dimensions
all_cells = ind_x + ind_y + ind_z
function_name = "nowrapNGP(dg, N, "
x_str = "iX"
y_str = "iY"
z_str = "iZ"
N_str = "N"
eval_strings = []
for cell in all_cells:
    cell_name = generate_cell_name(cell)
    eval_strings.append("double " + cell_name + " = " + function_name
                                  + x_str + (str(minval + cell[0]) if not cell[0] == 0 else "") + ", "
                                  + y_str + (str(minval + cell[1]) if not cell[1] == 0 else "") + ", "
                                  + z_str + (str(minval + cell[2]) if not cell[2] == 0 else "") + ");")

# Get rid of duplicates
eval_strings.sort()
unique_strings = set(eval_strings)

# Find all possible indices and wrap them only once
min_index, max_index = 0, 0
for cell in all_cells:
    min_index = min(min_index, np.array(cell).min())
    max_index = max(max_index, np.array(cell).max())

# Print the wrapped indices
print("")
print("/* Wrap the integer coordinates (not necessary for x) */")
for i in range(min_index, max_index + 1):
    if not i == 0:
        print("int iX" + str(i + minval) + " = " + x_str + (((" + " if i >= 0 else " - ") + str(abs(i))) if not i == 0 else "") + ";")
        print("int iY" + str(i + minval) + " = wrap(" + y_str + (((" + " if i >= 0 else " - ") + str(abs(i))) if not i == 0 else "") + ", " + N_str + ");")
        print("int iZ" + str(i + minval) + " = wrap(" + z_str + (((" + " if i >= 0 else " - ") + str(abs(i))) if not i == 0 else "") + ", " + N_str + ");")


# Print the cell evaluation strings
print("")
print("/* Retrieve the values necessary for the finite difference scheme */")
for es in unique_strings:
    print(es)

# Print the lines to accumulate the finite difference
print("")
print("/* Compute the finite difference along the x-axis */")
generate_accumulator("a[0]", ind_x, pos_x, coeff_x)
print("")
print("/* Compute the finite difference along the y-axis */")
generate_accumulator("a[1]", ind_y, pos_y, coeff_y)
print("")
print("/* Compute the finite difference along the z-axis */")
generate_accumulator("a[2]", ind_z, pos_z, coeff_z)

print("")
print("The number of evaluations is reduced by ", 100 * (1.0 - len(unique_strings) / len(eval_strings)), "%")
