/** Purpose of this test is to check that converting a mesh back to elements defined by point indices only works. This
 * is inverse to what is done with ìn the `test_from_points` test program.
 *
 * The way test program works is that it loads a mesh, which is then compared to point connectivity loaded from file.
 *
 * Test program takes 2 command line arguments:
 *
 * argv[1] : Path to the test file with element connectivity. Each line specifies zero-based indices of points of a
 * single element. It should be the be correct result to converting file from argv[1] into a CDUST
 * mesh.
 *
 * argv[2] : Path to CDUST mesh to load.
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
    unsigned *ptr1 = nullptr;
    unsigned size2 = 0, capacity2 = 0;
    unsigned *ptr2 = nullptr;

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

    char *read_mesh = read_file_to_string(path_mesh, 4096);
    TEST_ASSERT(read_mesh, "Could not read mesh to buffer");
    mesh_t mesh_comparison;
    real3_t *positions;
    int stat = deserialize_mesh(&mesh_comparison, &positions, read_mesh, &TEST_ALLOCATOR);
    free(read_mesh);
    free(positions);
    TEST_ASSERT(stat == 0, "Could not load comparison mesh");

    unsigned *flat_points;
    unsigned *point_counts;
    const unsigned n_elements = mesh_to_elements(&mesh_comparison, &point_counts, &flat_points, &TEST_ALLOCATOR);
    TEST_ASSERT(n_elements == mesh_comparison.n_surfaces, "Mesh conversion to elements failed.");
    TEST_ASSERT(n_elements == n_out, "Element count did not match.");
    TEST_ASSERT(memcmp(point_counts, n_per_element, sizeof(*point_counts) * n_elements) == 0, "Element point"
                                                                                              " counts did not match.");
    for (unsigned i = 0, j = 0; i < n_elements; ++i)
    {
        TEST_ASSERT(memcmp(flat_points + j, pts + j, sizeof(*pts) * point_counts[i]) == 0,
                    "Element %u did "
                    "not match.",
                    i);
        j += point_counts[i];
    }
    free(pts);
    free(n_per_element);
    free(flat_points);
    free(point_counts);

    mesh_free(&mesh_comparison, &TEST_ALLOCATOR);

    return 0;
}
