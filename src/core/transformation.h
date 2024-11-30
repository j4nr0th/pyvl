//
// Created by jan on 29.11.2024.
//

#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include "common.h"

typedef struct
{
    real3_t angles; // Rotation angles around x, y, and z axis
    real3_t offset; // Offsets by x, y, and z axis
} transformation_t;

#endif // TRANSFORMATION_H
