//
// Created by jan on 16.11.2024.
//

#include <errno.h>
#include <string.h>

#include "test_common.h"
#include "../src/io/mesh_io.h"



int main(int argc, char *argv[static restrict argc])
{
    if (argc != 3) return 1;
    const char *in_mesh_path = argv[1];
    const char *out_mesh_path = argv[2];

    FILE *f_in = fopen(in_mesh_path, "r");
    TEST_ASSERT(f_in, "Failed opening file %s, error %s", in_mesh_path, strerror(errno));

    enum {chunk_size = 1 << 12};
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

    mesh_t *msh = deserialize_mesh(buffer, &TEST_ALLOCATOR);
    TEST_ASSERT(msh, "Mesh not deserialized");
    free(buffer);

    char *const str_out = serialize_mesh(msh, &TEST_ALLOCATOR);
    TEST_ASSERT(str_out, "Mesh not serialized");
    mesh_free(msh, &TEST_ALLOCATOR);

    FILE *const f_out = fopen(out_mesh_path, "w");
    TEST_ASSERT(f_out, "Output file %s not open, %s", out_mesh_path, strerror(errno));

    const size_t len = strlen(str_out);
    TEST_ASSERT(fwrite(str_out, 1, len, f_out) == len, "Write not successful %s", strerror(errno));
    fclose(f_out);
    free(str_out);

    return 0;
}
