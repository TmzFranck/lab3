#include "image.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cstdint>

__global__ void grayscale_kernel(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height) {
	const auto x_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const auto y_pos = blockIdx.y * blockDim.y + threadIdx.y;

	if (x_pos >= width || y_pos >= height) {
		return;
	}

	const auto index = y_pos * width + x_pos;
	const auto pixel = input[index];

	const auto red = pixel.get_red_channel();
	const auto green = pixel.get_green_channel();
	const auto blue = pixel.get_blue_channel();

	const auto red_f = static_cast<float>(red);
	const auto green_f = static_cast<float>(green);
	const auto blue_f = static_cast<float>(blue);

	const auto gray_f = red_f * 0.2989F + green_f * 0.5870F + blue_f * 0.1140F;

	const auto gray = static_cast<std::uint8_t>(gray_f);
	const auto gray_pixel = Pixel<std::uint8_t>{ gray, gray, gray };

	output[index] = gray_pixel;
}

BitmapImage get_grayscale_cuda(const BitmapImage& source) {
	const auto height = source.get_height();
	const auto width = source.get_width();

	const auto raw_size = height * width * sizeof(Pixel<std::uint8_t>);

	auto gray_image = BitmapImage{ height, width };

	const auto* source_pixel = source.get_data();
	auto* destination_pixel = gray_image.get_data();

	Pixel<std::uint8_t>* source_pixel_dev = nullptr;
	Pixel<std::uint8_t>* destination_pixel_dev = nullptr;

	cudaMalloc((void**)&source_pixel_dev, raw_size);
	cudaMalloc((void**)&destination_pixel_dev, raw_size);

	cudaMemset(destination_pixel_dev, 0, raw_size);
	cudaMemcpy(source_pixel_dev, source_pixel, raw_size, cudaMemcpyHostToDevice);

	auto thread_dim = dim3{ 32, 32, 1 };
	auto block_dim = dim3{ divup(width, 32), divup(height, 32), 1 };

	grayscale_kernel << <thread_dim, block_dim, GRAYSCALE_SHARED_MEM >> > (source_pixel_dev, destination_pixel_dev, width, height);

	cudaMemcpy(destination_pixel, destination_pixel_dev, raw_size, cudaMemcpyDeviceToHost);

	cudaFree(source_pixel_dev);
	cudaFree(destination_pixel_dev);

	return gray_image;
}
