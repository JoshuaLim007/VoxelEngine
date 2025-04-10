# VoxelEngine
Realtime Voxel Raytracing Engine

Created using SDL2, CUDA, C++

Uses Brickmap structure based on this paper:
https://studenttheses.uu.nl/handle/20.500.12932/20460

This is in the early stages of development

Currently the basic raytracing algorithm is implemented: a brickmap acceleration structure for voxel raytracing. Memory is optimized to use bits instead of bytes (bit arrays). However the current project is missing key features listed below.


### ToDo list:
- Chunking
- PBR lighting
- LOD, further chunks with lower voxel resolution
- Proper Indirect lighting (denoise, temporal accumulation, etc...)
- Chunk data streaming
- Sky rendering
- Texture mapping and materials
- Better landscape generation
- Post process stack (Bloom, SSR, etc...)


### Screenshots:
![Demo Screenshot](demo/voxelrender0.png)
8k x 512 x 8k world size (no denoise)
![Demo Screenshot](demo/voxelrender1.png)
8k x 512 x 8k world size (no denoise)