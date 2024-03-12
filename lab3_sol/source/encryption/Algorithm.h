#pragma once

#include "encryption/FES.h"
#include "encryption/Key.h"
#include "image/bitmap_image.h"

#include <array>
#include <cstdint>
#include <exception>

enum class EncryptionStep : std::uint8_t {
	E = 0,
	D = 1,
	K = 2,
	T = 3
};

using EncryptionScheme = std::array<EncryptionStep, 16>;

[[nodiscard]] inline std::uint64_t encode(const EncryptionScheme& scheme) {
	auto encoded = std::uint64_t{ 0 };

	for (auto i = 0; i < 16; i++) {
		const auto step = scheme[i];
		encoded |= (static_cast<std::uint64_t>(step) << (i * 2));
	}

	encoded |= (encoded << 32);

	return encoded;
}

[[nodiscard]] inline EncryptionScheme decode(const std::uint64_t code) {
	auto decoded = EncryptionScheme{};

	const auto v1 = code & 4'294'967'295ULL;
	const auto v2 = code >> 32;

	if (v1 != v2) {
		throw std::exception{};
	}

	for (auto i = 0; i < 16; i++) {
		const auto val = (code >> (i * 2)) & 3;
		const auto step = static_cast<EncryptionStep>(val);
		decoded[i] = step;
	}

	return decoded;
}

[[nodiscard]] inline BitmapImage perform_scheme(const BitmapImage& original_image, const Key::key_type& key, const EncryptionScheme& scheme) {
	auto current_image = original_image;
	auto current_key = key;

	for (const auto step : scheme) {
		if (step == EncryptionStep::E) {
			current_image = FES::encrypt(current_image, current_key);
		}
		else if (step == EncryptionStep::D) {
			current_image = FES::decrypt(current_image, current_key);
		}
		else if (step == EncryptionStep::K) {
			current_key = Key::produce_new_key(current_key);
		}
		else if (step == EncryptionStep::T) {
			current_image = current_image.transpose();
		}
	}

	return current_image;
}

[[nodiscard]] EncryptionScheme retrieve_scheme(const std::uint64_t code);
