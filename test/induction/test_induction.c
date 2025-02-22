/** Purpose of this test is to verify that calculations of induction velocities are correct.
 * this ensures that entries of the induction matrix are correct.
 *
 */

#include "../../src/core/flow_solver.h"
#include "../test_common.h"

int main(int argc, char *CVL_ARRAY_ARG(argv, static restrict argc))
{
    /*
     * Test mesh with a single line and no surfaces.
     */
    /*
     * Test mesh with a single line and no surfaces.
     */
    const mesh_t TEST_MESH1 = {
        .n_points = 2,
        .n_lines = 1,
        .lines = (line_t[1]){{(geo_id_t){0, 0}, (geo_id_t){1, 0}}},
        .n_surfaces = 0,
        .surface_offsets = NULL,
        .surface_lines = NULL,
    };

    const real3_t positions[2] = {(real3_t){{-0.1, 0, 0.3}}, (real3_t){{1, 0, 0}}};
    const real3_t dir = line_direction(positions, &TEST_MESH1, (geo_id_t){0, 0});
    const real3_t v = {{0.2, 0.4, -0.3}};
    const real3_t induced = compute_mesh_line_induction(positions, v, (geo_id_t){0, 0}, &TEST_MESH1, 1e-6);

    //  Make sure that the induced velocity's direction is perpendicular to the d1 vector
    const real_t x = real3_dot(induced, dir);
    TEST_ASSERT(x < 1e-9 && x > -1e-6,
                "Dot product of induced velocity with line direction is not close to zero"
                ", but is instead %e!",
                x);

    const real3_t dr1 = real3_sub(v, positions[0]);
    const real3_t dr2 = real3_sub(v, positions[1]);

    const real3_t d = real3_unit(dir);
    const real_t tan_dist1 = real3_dot(dr1, d);
    const real_t tan_dist2 = real3_dot(dr2, d);
    const real_t normal_dist1 = sqrt(real3_dot(dr1, dr1) - tan_dist1 * tan_dist1);
    const real_t normal_dist2 = sqrt(real3_dot(dr2, dr2) - tan_dist2 * tan_dist2);
    TEST_ASSERT(normal_dist1 - normal_dist2 < 1e-9 && normal_dist1 - normal_dist2 > -1e-9,
                "Back to school for normal"
                " vectors, difference was %e!",
                normal_dist1 - normal_dist2);
    const real_t normal_dist = (normal_dist1 + normal_dist2) / 2.0;
    const real3_t real_induced =
        real3_mul1(real3_cross(v, d),
                   (atan2(tan_dist2, normal_dist) - atan2(tan_dist1, normal_dist)) * M_1_PI / 2.0 / normal_dist);

    const real_t ind_err = real3_max(real3_sub(real_induced, induced));
    TEST_ASSERT(ind_err < 1e-15 && ind_err > -1e-15, "Induction error was too large %e!", ind_err);

    return 0;
}
