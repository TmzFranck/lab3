#include "cuda/common.cuh"
#include "cuda/encryption.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cstdint>

bool test_three_a_adapter(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length, const unsigned int number_threads, const unsigned int number_blocks) {
	const auto raw_size = length * sizeof(std::uint64_t);

	std::uint64_t* values_dev = nullptr;
	std::uint64_t* hashes_dev = nullptr;

	cudaMalloc((void**)&values_dev, raw_size);
	cudaMalloc((void**)&hashes_dev, raw_size);

	cudaMemcpy(values_dev, values, raw_size, cudaMemcpyHostToDevice);

	auto thread_dim = dim3{ number_threads, 1, 1 };
	auto block_dim = dim3{ number_blocks, 1, 1 };

	hash << <block_dim, thread_dim, HASH_SHARED_MEM >> > (values_dev, hashes_dev, length);

	cudaMemcpy(hashes, hashes_dev, raw_size, cudaMemcpyDeviceToHost);

	cudaFree(values_dev);
	cudaFree(hashes_dev);

	cudaDeviceSynchronize();

	auto last_error = cudaGetLastError();

	return last_error == cudaSuccess;
}

bool test_three_b_adapter(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length, const unsigned int number_threads) {
	const auto raw_size = length * sizeof(std::uint64_t);

	std::uint64_t* values_dev = nullptr;
	std::uint64_t* hashes_dev = nullptr;

	cudaMalloc((void**)&values_dev, raw_size);
	cudaMalloc((void**)&hashes_dev, raw_size);

	cudaMemcpy(values_dev, values, raw_size, cudaMemcpyHostToDevice);

	auto thread_dim = dim3{ number_threads, 1, 1 };
	auto block_dim = dim3{ 1, 1, 1 };

	flat_hash << <block_dim, thread_dim, HASH_SHARED_MEM >> > (values_dev, hashes_dev, length);

	cudaMemcpy(hashes, hashes_dev, raw_size, cudaMemcpyDeviceToHost);

	cudaFree(values_dev);
	cudaFree(hashes_dev);

	cudaDeviceSynchronize();

	auto last_error = cudaGetLastError();

	return last_error == cudaSuccess;
}

bool test_three_c_adapter(const std::uint64_t* const values, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, const unsigned int number_threads, const unsigned int number_blocks) {
	std::uint64_t* values_dev = nullptr;
	unsigned int* indices_dev = nullptr;
	unsigned int* mem_cell_dev = nullptr;

	cudaMalloc((void**)&values_dev, length * sizeof(std::uint64_t));
	cudaMalloc((void**)&indices_dev, length * sizeof(unsigned int));
	cudaMalloc((void**)&mem_cell_dev, sizeof(unsigned int));

	cudaMemcpy(values_dev, values, length * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
	cudaMemset(indices_dev, 0, length * sizeof(unsigned int));
	cudaMemset(mem_cell_dev, 0, sizeof(unsigned int));

	auto thread_dim = dim3{ number_threads, 1, 1 };
	auto block_dim = dim3{ number_blocks, 1, 1 };

	find_hash << <block_dim, thread_dim, FIND_HASH_SHARED_MEM >> > (values_dev, indices_dev, length, searched_hash, mem_cell_dev);

	cudaMemcpy(indices, indices_dev, length * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(values_dev);
	cudaFree(indices_dev);
	cudaFree(mem_cell_dev);

	cudaDeviceSynchronize();

	auto last_error = cudaGetLastError();

	return last_error == cudaSuccess;
}

bool test_three_d_adapter(std::uint64_t* const hashes, const unsigned int length, const unsigned int number_threads, const unsigned int number_blocks) {
	const auto raw_size = length * sizeof(std::uint64_t);

	std::uint64_t* hashes_dev = nullptr;

	cudaMalloc((void**)&hashes_dev, raw_size);

	auto thread_dim = dim3{ number_threads, 1, 1 };
	auto block_dim = dim3{ number_blocks, 1, 1 };

	hash_schemes << <block_dim, thread_dim, HASH_SCHEMES_SHARED_MEM >> > (hashes_dev, length);

	cudaMemcpy(hashes, hashes_dev, raw_size, cudaMemcpyDeviceToHost);

	cudaFree(hashes_dev);

	cudaDeviceSynchronize();

	auto last_error = cudaGetLastError();

	return last_error == cudaSuccess;
}
