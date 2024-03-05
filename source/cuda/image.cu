#include "image.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdint>


__global__ void grayscale_kernel(const Pixel<std::uint8_t> *const input, Pixel<std::uint8_t> *const output, const unsigned int width, const unsigned int height) {
    // Berechnen Sie die Position des Pixels, das von diesem Thread verarbeitet wird
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Überprüfen Sie, ob die berechnete Position innerhalb der Grenzen des Bildes liegt
    if (x < width && y < height) {
        // Zugriff auf das Eingabepixel an der berechneten Position
        std::uint8_t* pixel_ptr = (std::uint8_t*)&input[y * width + x];

        // Berechnen Sie den Grauwert des Pixels
        std::uint8_t gray = pixel_ptr[0] * 0.2989 + pixel_ptr[1] * 0.5870 + pixel_ptr[2] * 0.1140;

        // Erstellen Sie ein neues Pixel mit dem berechneten Grauwert für alle Kanäle
        std::uint8_t* output_pixel_ptr = (std::uint8_t*)&output[y * width + x];
        output_pixel_ptr[0] = gray;
        output_pixel_ptr[1] = gray;
        output_pixel_ptr[2] = gray;
    }
}




BitmapImage get_grayscale_cuda(const BitmapImage& image) {
    // Holen Sie sich die Breite und Höhe des Eingabebildes
    const unsigned int width = image.get_width();
    const unsigned int height = image.get_height();

    // Erstellen Sie Speicher auf der GPU für das Eingabe- und Ausgabebild
    Pixel<std::uint8_t> *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(Pixel<std::uint8_t>));
    cudaMalloc(&d_output, width * height * sizeof(Pixel<std::uint8_t>));

    // Kopieren Sie das Eingabebild in den GPU-Speicher
    cudaMemcpy(d_input, image.get_data(), width * height * sizeof(Pixel<std::uint8_t>), cudaMemcpyHostToDevice);

    // Berechnen Sie die Anzahl der Blöcke und Threads
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(divup(width, threadsPerBlock.x), divup(height, threadsPerBlock.y));

    // Rufen Sie den Kernel auf
    grayscale_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

     // Warten Sie, bis alle Threads fertig sind
    cudaDeviceSynchronize();

    // Erstellen Sie ein neues BitmapImage-Objekt für das Ausgabebild
    BitmapImage output_image(height, width);

    // Kopieren Sie das Ausgabebild aus dem GPU-Speicher in das Ausgabebild-Objekt
    cudaMemcpy(output_image.get_data(), d_output, width * height * sizeof(Pixel<std::uint8_t>), cudaMemcpyDeviceToHost);

     // Geben Sie den GPU-Speicher frei
    cudaFree(d_input);
    cudaFree(d_output);

     // Geben Sie das Ausgabebild zurück
    return output_image;
}
