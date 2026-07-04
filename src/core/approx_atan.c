#include "approx_atan.h"
#include <math.h>

enum
{
    APPROX_ORDER = 7,
    APPROX_INTERVALS = 8,
};

const double atan2_approx_coeffs[APPROX_INTERVALS][APPROX_ORDER] = {
    {
        0.000000000000000000e+00,
        1.000000001835859686e+00,
        -2.620343334258834205e-07,
        -3.333182481503026984e-01,
        -4.280427433462487101e-04,
        2.063627159397426469e-01,
        -4.763655664570023290e-02,
    },
    {
        -1.097826515025740446e-07,
        1.000005586781660183e+00,
        -1.214499608589615783e-04,
        -3.318669160231263460e-01,
        -1.067586845690380454e-02,
        2.473101425773177042e-01,
        -1.205039307153051853e-01,
    },
    {
        -4.331579476224416447e-07,
        1.000018458270945709e+00,
        -3.238593747445859118e-04,
        -3.302206149751535369e-01,
        -1.806886148179876611e-02,
        2.648129198616193669e-01,
        -1.376454244930964688e-01,
    },
    {
        3.715352657464133455e-05,
        9.993198826612981200e-01,
        5.109704350131877119e-03,
        -3.528701751399468955e-01,
        3.532126797842084776e-02,
        1.973050573618351788e-01,
        -1.018601595043036462e-01,
    },
    {
        3.671956924712604649e-04,
        9.946678858057738104e-01,
        3.254666340358756960e-02,
        -4.395568510665789352e-01,
        1.900895365106952672e-01,
        4.923933314529738148e-02,
        -4.255451234096846447e-02,
    },
    {
        1.538208592016600132e-03,
        9.816764679055574039e-01,
        9.280868099049763287e-02,
        -5.891663616384894553e-01,
        3.997628019838609381e-01,
        -1.080419360612592666e-01,
        6.779443228575259862e-03,
    },
    {
        3.301656965760207951e-03,
        9.657305919660923443e-01,
        1.530476641743355737e-01,
        -7.108541747931319543e-01,
        5.383941803614084032e-01,
        -1.924867398110836625e-01,
        2.826478093499012328e-02,
    },
    {
        2.086098796852781448e-03,
        9.738673268928521587e-01,
        1.304744878408745090e-01,
        -6.776722641481709442e-01,
        5.111778712180228279e-01,
        -1.807011523007388820e-01,
        2.616579510795025501e-02,
    },

};

/**
 * Evaluate the approximation of atan2 in the range [0, 1]
 *
 * @param x The input value in the range [0, 1]
 * @return The approximate value of atan2(y, x) in the range [0, pi/4]
 */
static inline double atan2_approx_eval(const double x)
{
    const unsigned interval = (unsigned)(x * APPROX_INTERVALS);
    const double *restrict coeffs = atan2_approx_coeffs[interval];
    // Manually unrolled FMA
    double v = fma(coeffs[6], x, coeffs[5]);
    v = fma(v, x, coeffs[4]);
    v = fma(v, x, coeffs[3]);
    v = fma(v, x, coeffs[2]);
    v = fma(v, x, coeffs[1]);
    v = fma(v, x, coeffs[0]);

    return v;
}

typedef enum
{
    QUADRANT_1 = 0, // x >= 0, y >= 0
    QUADRANT_2 = 1, // x < 0, y >= 0
    QUADRANT_4 = 2, // x >= 0, y < 0
    QUADRANT_3 = 3, // x < 0, y < 0
    X_NEGATIVE_MASK = 1,
    Y_NEGATIVE_MASK = 2,
} atan2_quadrant_t;

double atan2_approx(const double y, const double x)
{
    // Determine the quadrant of the point (x, y)
    atan2_quadrant_t quadrant = 0;
    double abs_x, abs_y;
    if (signbit(x))
    {
        quadrant = X_NEGATIVE_MASK;
        abs_x = -x;
    }
    else
    {
        abs_x = x;
    }
    if (signbit(y))
    {
        quadrant |= Y_NEGATIVE_MASK;
        abs_y = -y;
    }
    else
    {
        abs_y = y;
    }

    double v;
    if (abs_x > abs_y)
    {
        // Use the approximation for atan2(y, x) = atan(y/x)
        v = atan2_approx_eval(abs_y / abs_x);
    }
    else if (abs_x == abs_y)
    {
        if (abs_x == 0) // means abs_y == 0 as well
        {
            switch (quadrant)
            {
            case QUADRANT_1:
                return 0.0;
            case QUADRANT_2:
                return M_PI / 2;
            case QUADRANT_3:
                return M_PI;
            case QUADRANT_4:
                return -M_PI / 2;
            }
        }
        v = M_PI / 4;
    }
    else
    {
        // Use the approximation for atan2(y, x) = pi/2 - atan(x/y)
        v = M_PI / 2 - atan2_approx_eval(abs_x / abs_y);
    }

    // Adjust the result based on the quadrant
    // switch (quadrant)
    // {
    // case QUADRANT_2:
    //     return M_PI - v;
    // case QUADRANT_3:
    //     return -(M_PI - v);
    // case QUADRANT_1:
    //     return v;
    // case QUADRANT_4:
    //     return -v;
    // }
    if (quadrant & X_NEGATIVE_MASK)
    {
        v = M_PI - v;
    }
    if (quadrant & Y_NEGATIVE_MASK)
    {
        v = -v;
    }

    return v;
}

double atan_approx(const double x)
{
    // Quick path
    if (x == 0)
        return 0;

    bool is_negative = signbit(x);
    const double abs_x = fabs(x);

    double v;
    if (abs_x < 1)
    {
        // Use the approximation for atan(y / x)
        v = atan2_approx_eval(abs_x);
    }
    else if (abs_x == 1)
    {
        v = M_PI / 4;
    }
    else
    {
        // Use the approximation for atan(y / x) = pi/2 - atan(x / y)
        v = M_PI / 2 - atan2_approx_eval(1 / abs_x);
    }

    // Adjust the result based on the quadrant
    if (is_negative)
    {
        v = -v;
    }

    return v;
}
