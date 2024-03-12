#pragma once

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

enum class ChannelOrder {
	BGR,
	BRG,
	GBR,
	GRB,
	RBG,
	RGB
};

template<typename channel_type>
class Pixel {
public:
	using value_type = channel_type;

	static ChannelOrder channel_order;

	__host__ __device__ Pixel() noexcept = default;

	__host__ __device__ Pixel(const channel_type red, const channel_type green, const channel_type blue) noexcept
		: red_channel{ red }, green_channel{ green }, blue_channel{ blue } {
	}

	__host__ __device__ [[nodiscard]] channel_type get_red_channel() const noexcept {
		return red_channel;
	}

	__host__ __device__ [[nodiscard]] channel_type get_green_channel() const noexcept {
		return green_channel;
	}

	__host__ __device__ [[nodiscard]] channel_type get_blue_channel() const noexcept {
		return blue_channel;
	}

	[[nodiscard]] Pixel operator^(const Pixel& other) const noexcept {
		const channel_type new_red = red_channel ^ other.red_channel;
		const channel_type new_green = green_channel ^ other.green_channel;
		const channel_type new_blue = blue_channel ^ other.blue_channel;

		return Pixel{ new_red, new_green, new_blue };
	}

	[[nodiscard]] bool operator==(const Pixel& other) const noexcept {
		return red_channel == other.red_channel && green_channel == other.green_channel && blue_channel == other.blue_channel;
	}

private:
	channel_type red_channel{};
	channel_type green_channel{};
	channel_type blue_channel{};
};

template<typename channel_type>
ChannelOrder Pixel<channel_type>::channel_order = ChannelOrder::RGB;
