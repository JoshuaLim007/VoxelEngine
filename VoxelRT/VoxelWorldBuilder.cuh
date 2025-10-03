#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_math.h"
#include "DDA.cuh"

using namespace GPUDDA;

__global__ void PopulateVoxels(BitArray voxels, uint3 size);

