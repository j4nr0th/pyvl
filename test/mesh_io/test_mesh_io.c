/** Test checking the mesh serialization/deserialization works symmetrically.
 *
 * This test program will load a mesh from a file, convert it into a string, which can optionally
 * be written back to an output file to check the result, then compare that the produced string exactly
 * matches the input string. This includes the comments and all whitespace, so manually check the
 * generated file is correct before feeding it into the test.
 *
 * Test program takes 1 or 2 command line arguments:
 *
 * argv[1] : Path to test file to load. It should be a file generated by CDUST, which was
 *     manually checked to be correct.
 *
 * argv[2] (optional): Path to output location where the mesh, which was loaded from argv[1], will be
 *     stored. This is to allow for easier inspection of files.
 *
 */

#include <errno.h>
#include <string.h>

#include "../../src/core/mesh_io.h"
#include "../test_common.h"

int main(int argc, char *argv[static restrict argc])
{
    if (argc < 2 || argc > 3)
        return 1;
    const char *in_mesh_path = argv[1];
    const char *out_mesh_path = argc == 3 ? argv[2] : nullptr;

    FILE *f_in = fopen(in_mesh_path, "r");
    TEST_ASSERT(f_in, "Failed opening file %s, error %s", in_mesh_path, strerror(errno));

    enum
    {
        chunk_size = 1 << 12
    };
    size_t buffer_size = chunk_size;
    char *buffer = malloc((sizeof *buffer) * buffer_size);
    TEST_ASSERT(buffer, "Buffer not allocate");
    size_t read_sz;
    while ((read_sz = fread(buffer + buffer_size - chunk_size, 1, chunk_size, f_in)) == chunk_size)
    {
        buffer_size += chunk_size;
        char *new_buffer = realloc(buffer, buffer_size * sizeof(*buffer));
        TEST_ASSERT(new_buffer, "Buffer not reallocated");
        buffer = new_buffer;
    }
    fclose(f_in);
    buffer[buffer_size - chunk_size + read_sz] = 0;

    mesh_t msh;
    real3_t *positions;
    int stat = deserialize_mesh(&msh, &positions, buffer, &TEST_ALLOCATOR);
    TEST_ASSERT(stat == 0, "Mesh not deserialized");

    char *const str_out = serialize_mesh(&msh, positions, &TEST_ALLOCATOR);
    TEST_ASSERT(str_out, "Mesh not serialized");
    mesh_free(&msh, &TEST_ALLOCATOR);
    TEST_ALLOCATOR.deallocate(TEST_ALLOCATOR.state, positions);

    const size_t len = strlen(str_out);
    if (out_mesh_path)
    {
        FILE *const f_out = fopen(out_mesh_path, "w");
        TEST_ASSERT(f_out, "Output file %s not open, %s", out_mesh_path, strerror(errno));
        TEST_ASSERT(fwrite(str_out, 1, len, f_out) == len, "Write not successful %s", strerror(errno));
        fclose(f_out);
    }

    const int cmp_res = memcmp(buffer, str_out, len);
    TEST_ASSERT(cmp_res == 0, "Two buffers did not compare as equal (memcmp was %d).", cmp_res);
    free(buffer);
    free(str_out);

    return 0;
}
