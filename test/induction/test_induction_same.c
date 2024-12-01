/** Purpose of this test is to verify that calculations of induction velocities are correct.
 * this ensures that entries of the induction matrix are correct.
 *
 */

#include "../../src/core/flow_solver.h"
#include "../../src/core/mesh_io.h"
#include "../test_common.h"

constexpr real_t TOL = 1e-12;

int main(int argc, char *argv[static argc])
{
    TEST_ASSERT(argc == 2, "Program requires 1 command line argument, but %d were given.", argc - 1);
    const char *in_mesh_path = argv[1];

    enum
    {
        chunk_size = 1 << 12
    };
    char *buffer = read_file_to_string(in_mesh_path, chunk_size);

    mesh_t *msh = deserialize_mesh(buffer, &TEST_ALLOCATOR);
    TEST_ASSERT(msh, "Mesh not deserialized");
    free(buffer);

    real3_t *const surface_centers = malloc(msh->n_surfaces * sizeof(*surface_centers));
    TEST_ASSERT(surface_centers, "Did not allocate surface center memory.");
    real3_t *const line_buffer = malloc(msh->n_lines * msh->n_surfaces * sizeof(*line_buffer));
    TEST_ASSERT(line_buffer, "Did not allocate line buffer memory.");
    real3_t *const surface_inductions = malloc(msh->n_surfaces * msh->n_surfaces * sizeof(*surface_inductions));
    TEST_ASSERT(surface_inductions, "Did not allocate surface inductions memory.");

    for (unsigned i_surf = 0; i_surf < msh->n_surfaces; ++i_surf)
    {
        surface_centers[i_surf] = surface_center(msh, (geo_id_t){.value = i_surf});
    }

    compute_line_induction(msh->n_lines, msh->lines, msh->n_points, msh->positions, msh->n_surfaces, surface_centers,
                           line_buffer, TOL);

    for (unsigned i_cp = 0; i_cp < msh->n_surfaces; ++i_cp)
    {
        for (unsigned i_ln = 0; i_ln < msh->n_lines; ++i_ln)
        {
            const real3_t ind1 =
                compute_mesh_line_induction(surface_centers[i_cp], (geo_id_t){.value = i_ln}, msh, TOL);
            TEST_ASSERT(real3_mag(real3_sub(ind1, line_buffer[i_cp * msh->n_lines + i_ln])) < TOL,
                        "Difference between the two inductions computed is large: %e.",
                        real3_mag(real3_sub(ind1, line_buffer[i_cp * msh->n_lines + i_ln])));
        }
    }
    line_induction_to_surface_induction(msh->n_surfaces, msh->surfaces, msh->n_lines, msh->n_surfaces, line_buffer,
                                        surface_inductions);
    for (unsigned i_cp = 0; i_cp < msh->n_surfaces; ++i_cp)
    {
        for (unsigned i_surf = 0; i_surf < msh->n_surfaces; ++i_surf)
        {
            const real3_t ind1 =
                compute_mesh_surface_induction(surface_centers[i_cp], (geo_id_t){.value = i_surf}, msh, TOL);
            TEST_ASSERT(real3_mag(real3_sub(ind1, surface_inductions[i_cp * msh->n_surfaces + i_surf])) < TOL,
                        "Difference between the two inductions computed is large: %e.",
                        real3_mag(real3_sub(ind1, surface_inductions[i_cp * msh->n_surfaces + i_surf])));
        }
    }

    free(surface_inductions);
    free(line_buffer);
    free(surface_centers);
    mesh_free(msh, &TEST_ALLOCATOR);

    return 0;
}
