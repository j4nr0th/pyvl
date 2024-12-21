/** Test checking the mesh serialization/deserialization works symmetrically.
 *
 * This test program will load a mesh from a file, compute its dual mesh, which it will then compare against a mesh
 * it loads from a manually checked output file and ensure the computed mesh is identical, except for positions array,
 * since that's made of floats, and it's not really prescribed what it should really be anyway for a dual mesh.
 *
 * Test program takes 2 or 3 command line arguments:
 *
 * argv[1] : Path to test file to load. It should be a file generated by CDUST, which was
 *     manually checked to be correct.
 *
 * argv[2] : Path to test file to load. It should be a file generated by CDUST, which was
 *     manually checked to be correct.
 *
 * argv[3] (optional): Path to output location where the mesh, which was loaded from argv[1], will be
 *     stored. This is to allow for easier inspection of files.
 *
 */

#include <errno.h>
#include <string.h>

#include "../../src/core/mesh.h"
#include "../../src/core/mesh_io.h"
#include "../test_common.h"

int main(int argc, char *argv[static restrict argc])
{
    TEST_ASSERT(argc == 3 || argc == 4, "Wrong number of parameters %d", argc);
    const char *in_mesh_path = argv[1];
    const char *cmp_mesh_path = argv[2];

    enum
    {
        chunk_size = 1 << 12
    };
    char *buffer = read_file_to_string(in_mesh_path, chunk_size);
    int stat;

    mesh_t msh;
    real3_t *positions;
    stat = deserialize_mesh(&msh, &positions, buffer, &TEST_ALLOCATOR);
    TEST_ASSERT(stat == 0, "Mesh not deserialized");
    free(buffer);

    mesh_t dual;
    stat = mesh_dual_from_primal(&dual, &msh, &TEST_ALLOCATOR);
    TEST_ASSERT(stat == 0, "Dual failed");
    real3_t *surface_centers = malloc(sizeof *surface_centers * msh.n_surfaces);
    TEST_ASSERT(surface_centers, "Failed to allocate center buffer");
    for (unsigned i = 0; i < msh.n_surfaces; ++i)
    {
        surface_centers[i] = surface_center(positions, &msh, (geo_id_t){.value = i});
    }

    mesh_free(&msh, &TEST_ALLOCATOR);
    free(positions);

    if (argc == 4)
    {
        char *const str_out = serialize_mesh(&dual, surface_centers, &TEST_ALLOCATOR);
        TEST_ASSERT(str_out, "Mesh not serialized");
        const size_t len = strlen(str_out);
        FILE *const f_out = fopen(argv[3], "w");
        TEST_ASSERT(f_out, "Output file %s not open, %s", argv[3], strerror(errno));
        TEST_ASSERT(fwrite(str_out, 1, len, f_out) == len, "Write not successful %s", strerror(errno));
        fclose(f_out);
        free(str_out);
    }

    buffer = read_file_to_string(cmp_mesh_path, chunk_size);
    mesh_t cmp;
    stat = deserialize_mesh(&cmp, &positions, buffer, &TEST_ALLOCATOR);
    TEST_ASSERT(stat == 0, "Comparison not deserialized");
    free(buffer);

    TEST_ASSERT(cmp.n_points == dual.n_points, "Point counts do not match: %u vs %u", cmp.n_points, dual.n_points);
    TEST_ASSERT(cmp.n_lines == dual.n_lines, "Line counts do not match: %u vs %u", cmp.n_lines, dual.n_lines);
    TEST_ASSERT(cmp.n_surfaces == dual.n_surfaces, "Surface counts do not match: %u vs %u", cmp.n_surfaces,
                dual.n_surfaces);

    TEST_ASSERT(memcmp(cmp.lines, dual.lines, sizeof(*cmp.lines) * cmp.n_lines) == 0,
                "Comparison of line arrays failed.");
    TEST_ASSERT(
        memcmp(cmp.surface_offsets, dual.surface_offsets, sizeof(*cmp.surface_offsets) * (cmp.n_surfaces + 1)) == 0,
        "Comparison of surfaces.");
    TEST_ASSERT(memcmp(cmp.surface_lines, dual.surface_lines,
                       sizeof(*cmp.surface_lines) * cmp.surface_offsets[cmp.n_surfaces]) == 0,
                "Comparison of surfaces.");

    mesh_free(&cmp, &TEST_ALLOCATOR);
    mesh_free(&dual, &TEST_ALLOCATOR);
    free(surface_centers);
    free(positions);

    return 0;
}
