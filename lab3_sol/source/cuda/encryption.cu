#include "encryption.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <vector>

unsigned int divup(const unsigned int numerator, const unsigned int denominator)
 {
	auto res = numerator / denominator;
	if (numerator % denominator != 0) {
		res++;
	}

	return res;
}

__device__ std::uint64_t hash_one_value(const std::uint64_t value) {
	constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
	constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
	constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

	const auto val_1 = (value >> 14) + val_a;
	const auto val_2 = (value << 54) ^ val_b;
	const auto val_3 = (val_1 + val_2) << 4;
	const auto val_4 = (value % val_c) * 137;

	const auto final_hash = val_3 ^ val_4;

	return final_hash;
}

__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {
	const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= length) {
		return;
	}

	const auto value = values[thread_idx];
	const auto final_hash = hash_one_value(value);
	hashes[thread_idx] = final_hash;
}

__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {
	const auto number_threads = blockDim.x;
	const auto thread_idx = blockIdx.x * number_threads + threadIdx.x;

	if (thread_idx >= length) {
		return;
	}

	for (unsigned int index = thread_idx; index < length; index += number_threads) {
		const auto value = values[index];
		const auto final_hash = hash_one_value(value);
		hashes[index] = final_hash;
	}
}

__global__ void find_hash(const std::uint64_t* const hashes, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, unsigned int* const ptr) {
	const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= length) {
		return;
	}

	const auto hash = hashes[thread_idx];
	if (hash != searched_hash) {
		return;
	}

	const auto index_to_write_to = atomicAdd(ptr, 1U);
	indices[index_to_write_to] = thread_idx;
}

__global__ void hash_schemes(std::uint64_t* const hashes, const unsigned int length) {
	const auto number_threads = gridDim.x * blockDim.x;
	const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= length) {
		return;
	}

	for (unsigned int index = thread_idx; index < length; index += number_threads) {
		const auto protoscheme = std::uint64_t{ index };
		const auto scheme = protoscheme | (protoscheme << 32);

		const auto hash = hash_one_value(scheme);
		hashes[index] = hash;
	}
}

std::uint64_t retrieve_scheme_adapter(const std::uint64_t code) {
	const auto num_vals = 1024U * 1024U;

	std::uint64_t* hashes_dev;
	cudaMalloc((void**)&hashes_dev, num_vals * sizeof(std::uint64_t));

	hash_schemes << <1024, 1024 >> > (hashes_dev, num_vals);

	unsigned int* indices_dev;
	cudaMalloc((void**)&indices_dev, num_vals * sizeof(unsigned int));
	cudaMemset(indices_dev, 0, num_vals * sizeof(unsigned int));

	unsigned int* mem_cell_dev;
	cudaMalloc((void**)&mem_cell_dev, sizeof(unsigned int));
	cudaMemset(mem_cell_dev, 0, sizeof(unsigned int));

	find_hash << <1024, 1024 >> > (hashes_dev, indices_dev, num_vals, code, mem_cell_dev);

	auto indices = std::vector<unsigned int>(num_vals);

	cudaMemcpy(indices.data(), indices_dev, num_vals * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(mem_cell_dev);
	cudaFree(indices_dev);
	cudaFree(hashes_dev);

	const auto protoscheme = std::uint64_t{ indices[0] };
	const auto scheme = protoscheme | (protoscheme << 32);

	return scheme;
}
