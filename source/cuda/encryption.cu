#include "encryption.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {
    constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
    constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
    constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

    const auto bx = blockIdx.x;
    const auto tx = threadIdx.x;

    for (std::uint64_t k = bx * blockDim.x + tx; k < length; k += blockDim.x * gridDim.x) {
        const auto value = values[k];

        const auto val_1 = (value >> 14) + val_a;
        const auto val_2 = (value << 54) ^ val_b;
        const auto val_3 = (val_1 + val_2) << 4;
        const auto val_4 = (value % val_c) * 137;

        const auto final_hash = val_3 ^ val_4;
        hashes[k] = final_hash;
    }
}

__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {
    constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
    constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
    constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

    const auto tx = threadIdx.x;

    for (std::uint64_t k = tx; k < length; k += blockDim.x) {
        const auto value = values[k];

        const auto val_1 = (value >> 14) + val_a;
        const auto val_2 = (value << 54) ^ val_b;
        const auto val_3 = (val_1 + val_2) << 4;
        const auto val_4 = (value % val_c) * 137;

        const auto final_hash = val_3 ^ val_4;
        hashes[k] = final_hash;
    }
}

__global__ void find_hash(const std::uint64_t* const hashes, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, unsigned int* const ptr) {
    const auto global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index < length) {
        const auto hash = hashes[global_index];

        if (hash == searched_hash) {
            indices[global_index] = global_index;
            atomicMin(ptr, global_index);
        }
    }
}

__global__ void hash_schemes(std::uint64_t* const hashes, const unsigned int length) {
    const auto global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index < length) {
        // Generate the encoded scheme
        std::uint64_t scheme = 0;
        for (int i = 0; i < 16; ++i) {
            scheme |= static_cast<std::uint64_t>(global_index >> (i * 2) & 3) << (i * 2);
        }

        // Replicate the lower 32 bits into the upper 32 bits
        scheme |= scheme << 32;

        // Store the encoded scheme in the hashes array
        hashes[global_index] = scheme;
    }
}