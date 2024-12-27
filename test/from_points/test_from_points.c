/** Purpose of this test is to check that creating a mesh from elements generates correct connectivity for the mesh.
 * This test program is especially important, since most unstructured meshes store the connectivity just connectivity
 * per element.
 *
 * The way test program works is that it loads point connectivity, from which a mesh is built, then compared to another
 * one, which is loaded from a file and should be the correct solution.
 *
 * Test program takes 2 command line arguments:
 *
 * argv[1] : Path to the test file with element connectivity. Each line specifies zero-based indices of points of a
 * single element.
 *
 * argv[2] : Path to test file to load. It should be the be correct result to converting file from argv[1] into a CVL
 * mesh.
 *
 */

#include <ctype.h>
#include <string.h>

#include "../../src/core/mesh_io.h"
#include "../test_common.h"

static void push_buffer(unsigned value, unsigned *size, unsigned *capacity, unsigned **buffer)
{
    if (*size == *capacity)
    {
        const unsigned new_capacity = *capacity > 0 ? *capacity << 1 : 64;

        unsigned *const new_ptr = realloc(*buffer, sizeof(**buffer) * new_capacity);
        TEST_ASSERT(new_ptr, "Failed reallocating buffer to size %u", new_capacity);
        *buffer = new_ptr;
        *capacity = new_capacity;
    }
    *(*buffer + *size) = value;
    *size = (*size) + 1;
}

static void load_points(const char *path, unsigned *n_out, unsigned **n_per_element, unsigned **pts_out)
{
    char *const str = read_file_to_string(path, 4096);
    TEST_ASSERT(str, "Could not read file %s", path);
    unsigned size1 = 0, capacity1 = 0;
    unsigned *ptr1 = NULL;
    unsigned size2 = 0, capacity2 = 0;
    unsigned *ptr2 = NULL;

    char *p = str;
    unsigned current_cnt = 0;
    while (*p)
    {
        char *end;
        const unsigned v = strtoul(p, &end, 10);
        TEST_ASSERT(end != p, "Failed parsing from \"%s", p);
        p = end;
        push_buffer(v, &size2, &capacity2, &ptr2);
        current_cnt += 1;
        while (isspace(*p))
        {
            if (*p == '\n' && current_cnt != 0)
            {
                push_buffer(current_cnt, &size1, &capacity1, &ptr1);
                current_cnt = 0;
            }
            p += 1;
        }
    }
    if (*p == '\n' && current_cnt != 0)
    {
        push_buffer(current_cnt, &size1, &capacity1, &ptr1);
    }

    *n_out = size1;
    *n_per_element = ptr1;
    *pts_out = ptr2;

    free(str);
}

int main(int argc, char *argv[static argc])
{
    TEST_ASSERT(argc == 3, "Not the correct number of arguments");
    const char *path_points = argv[1];
    const char *path_mesh = argv[2];

    unsigned n_out, *n_per_element, *pts;
    load_points(path_points, &n_out, &n_per_element, &pts);
    // {
    //     unsigned *p = pts;
    //     for (unsigned i = 0; i < n_out; ++i)
    //     {
    //         printf("Surface %u:", i);
    //         for (unsigned j = 0; j < n_per_element[i]; ++j)
    //         {
    //             printf(" %u", *p);
    //             ++p;
    //         }
    //         printf("\n");
    //     }
    // }

    char *read_mesh = read_file_to_string(path_mesh, 4096);
    TEST_ASSERT(read_mesh, "Could not read mesh to buffer");
    mesh_t mesh_comparison;
    real3_t *positions;
    int stat = deserialize_mesh(&mesh_comparison, &positions, read_mesh, &TEST_ALLOCATOR);
    free(read_mesh);
    free(positions);
    TEST_ASSERT(stat == 0, "Could not load comparison mesh");

    mesh_t mesh;
    stat = mesh_from_elements(&mesh, n_out, n_per_element, pts, &TEST_ALLOCATOR);
    TEST_ASSERT(stat == 0, "Failed computing the mesh");
    free(n_per_element);
    free(pts);

    TEST_ASSERT(mesh.n_lines == mesh_comparison.n_lines, "Line counts do not match: %u vs %u", mesh.n_lines,
                mesh_comparison.n_lines);
    TEST_ASSERT(mesh.n_surfaces == mesh_comparison.n_surfaces, "Surface counts do not match: %u vs %u", mesh.n_surfaces,
                mesh_comparison.n_surfaces);

    TEST_ASSERT(memcmp(mesh.lines, mesh_comparison.lines, sizeof(*mesh.lines) * mesh.n_lines) == 0,
                "Comparison of line arrays failed.");

    TEST_ASSERT(memcmp(mesh.surface_offsets, mesh_comparison.surface_offsets,
                       sizeof(*mesh.surface_offsets) * mesh.n_surfaces) == 0,
                "Comparison of surface offsets.");
    TEST_ASSERT(memcmp(mesh.surface_lines, mesh_comparison.surface_lines,
                       sizeof(*mesh.surface_lines) * mesh.surface_offsets[mesh.n_surfaces]) == 0,
                "Comparison of surface lines.");

    mesh_free(&mesh, &TEST_ALLOCATOR);
    mesh_free(&mesh_comparison, &TEST_ALLOCATOR);

    return 0;
}
