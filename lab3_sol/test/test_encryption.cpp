#include "test.h"

#include "encryption/Algorithm.h"
#include "encryption/FES.h"
#include "encryption/Key.h"
#include "image/bitmap_image.h"

#include <array>
#include <type_traits>

class EncryptionTest : public LabTest {};

TEST_F(EncryptionTest, test_one_a_one) {
	auto step_e = EncryptionStep::E;
	auto step_d = EncryptionStep::D;
	auto step_k = EncryptionStep::K;
	auto step_t = EncryptionStep::T;

	static_assert(std::is_same_v<std::array<EncryptionStep, 16>, EncryptionScheme>);
}

TEST_F(EncryptionTest, test_one_b_one) {
	auto scheme_1 = EncryptionScheme{};
	scheme_1[0] = EncryptionStep::E;
	scheme_1[1] = EncryptionStep::D;
	scheme_1[2] = EncryptionStep::D;
	scheme_1[3] = EncryptionStep::K;
	scheme_1[4] = EncryptionStep::K;
	scheme_1[5] = EncryptionStep::T;
	scheme_1[6] = EncryptionStep::E;
	scheme_1[7] = EncryptionStep::T;
	scheme_1[8] = EncryptionStep::T;
	scheme_1[9] = EncryptionStep::T;
	scheme_1[10] = EncryptionStep::K;
	scheme_1[11] = EncryptionStep::K;
	scheme_1[12] = EncryptionStep::E;
	scheme_1[13] = EncryptionStep::D;
	scheme_1[14] = EncryptionStep::E;
	scheme_1[15] = EncryptionStep::D;

	const auto expected_code_1 = 0b01000100'10101111'11001110'10010100'01000100'10101111'11001110'10010100ULL;
	const auto code_1 = encode(scheme_1);

	ASSERT_EQ(code_1, expected_code_1);

	auto scheme_2 = EncryptionScheme{};
	scheme_2[0] = EncryptionStep::K;
	scheme_2[1] = EncryptionStep::T;
	scheme_2[2] = EncryptionStep::K;
	scheme_2[3] = EncryptionStep::T;
	scheme_2[4] = EncryptionStep::E;
	scheme_2[5] = EncryptionStep::D;
	scheme_2[6] = EncryptionStep::E;
	scheme_2[7] = EncryptionStep::D;
	scheme_2[8] = EncryptionStep::K;
	scheme_2[9] = EncryptionStep::T;
	scheme_2[10] = EncryptionStep::K;
	scheme_2[11] = EncryptionStep::T;
	scheme_2[12] = EncryptionStep::E;
	scheme_2[13] = EncryptionStep::D;
	scheme_2[14] = EncryptionStep::E;
	scheme_2[15] = EncryptionStep::D;

	const auto expected_code_2 = 0b01000100'11101110'01000100'11101110'01000100'11101110'01000100'11101110ULL;
	const auto code_2 = encode(scheme_2);

	ASSERT_EQ(code_2, expected_code_2);
}

TEST_F(EncryptionTest, test_one_c_one) {
	auto expected_scheme_1 = EncryptionScheme{};
	expected_scheme_1[0] = EncryptionStep::E;
	expected_scheme_1[1] = EncryptionStep::D;
	expected_scheme_1[2] = EncryptionStep::D;
	expected_scheme_1[3] = EncryptionStep::K;
	expected_scheme_1[4] = EncryptionStep::K;
	expected_scheme_1[5] = EncryptionStep::T;
	expected_scheme_1[6] = EncryptionStep::E;
	expected_scheme_1[7] = EncryptionStep::T;
	expected_scheme_1[8] = EncryptionStep::T;
	expected_scheme_1[9] = EncryptionStep::T;
	expected_scheme_1[10] = EncryptionStep::K;
	expected_scheme_1[11] = EncryptionStep::K;
	expected_scheme_1[12] = EncryptionStep::E;
	expected_scheme_1[13] = EncryptionStep::D;
	expected_scheme_1[14] = EncryptionStep::E;
	expected_scheme_1[15] = EncryptionStep::D;

	const auto code_1 = 0b01000100'10101111'11001110'10010100'01000100'10101111'11001110'10010100ULL;
	const auto scheme_1 = decode(code_1);

	ASSERT_EQ(scheme_1, expected_scheme_1);

	auto expected_scheme_2 = EncryptionScheme{};
	expected_scheme_2[0] = EncryptionStep::T;
	expected_scheme_2[1] = EncryptionStep::K;
	expected_scheme_2[2] = EncryptionStep::T;
	expected_scheme_2[3] = EncryptionStep::K;
	expected_scheme_2[4] = EncryptionStep::D;
	expected_scheme_2[5] = EncryptionStep::E;
	expected_scheme_2[6] = EncryptionStep::D;
	expected_scheme_2[7] = EncryptionStep::E;
	expected_scheme_2[8] = EncryptionStep::T;
	expected_scheme_2[9] = EncryptionStep::K;
	expected_scheme_2[10] = EncryptionStep::T;
	expected_scheme_2[11] = EncryptionStep::K;
	expected_scheme_2[12] = EncryptionStep::D;
	expected_scheme_2[13] = EncryptionStep::E;
	expected_scheme_2[14] = EncryptionStep::D;
	expected_scheme_2[15] = EncryptionStep::E;

	const auto code_2 = 0b00010001'10111011'00010001'10111011'00010001'10111011'00010001'10111011ULL;
	const auto scheme_2 = decode(code_2);

	ASSERT_EQ(scheme_2, expected_scheme_2);
}

TEST_F(EncryptionTest, test_one_c_two) {
	const auto code_1 = 0b11110100'11111010'10100101'01010000'11110000'11111010'10100101'01010000ULL;
	const auto code_2 = 0b11000000'00111110'10101001'01011101'11000000'00111110'10101001'01010101ULL;
	const auto code_3 = 0b11110000'01101000'10100101'01010000'11110000'01101010'10100101'01010000ULL;
	const auto code_4 = 0b11110000'11111010'10100101'01010010'11110000'11111010'10100101'01010000ULL;

	ASSERT_THROW(const auto val = decode(code_1), std::exception);
	ASSERT_THROW(const auto val = decode(code_2), std::exception);
	ASSERT_THROW(const auto val = decode(code_3), std::exception);
	ASSERT_THROW(const auto val = decode(code_4), std::exception);
}

TEST_F(EncryptionTest, test_one_d_one) {
	const auto get_image = []() {
		auto image = BitmapImage{ 480, 960 };

		// Who knows how they index their pixels
		try {
			for (auto y = 0; y < 480; y++) {
				for (auto x = 0; x < 960; x++) {
					const auto pixel = BitmapImage::BitmapPixel{ static_cast<std::uint8_t>(x & 255), static_cast<std::uint8_t>((x * y) & 255), static_cast<std::uint8_t>((x + y * y * x) & 255) };
					image.set_pixel(y, x, pixel);
				}
			}
		}
		catch (std::exception ex) {
			for (auto y = 0; y < 480; y++) {
				for (auto x = 0; x < 960; x++) {
					const auto pixel = BitmapImage::BitmapPixel{ static_cast<std::uint8_t>(x & 255), static_cast<std::uint8_t>((x * y) & 255), static_cast<std::uint8_t>((x + y * y * x) & 255) };
					image.set_pixel(x, y, pixel);
				}
			}
		}

		return image;

		};

	const auto get_scheme = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::E;
		scheme[1] = EncryptionStep::D;
		scheme[2] = EncryptionStep::D;
		scheme[3] = EncryptionStep::K;
		scheme[4] = EncryptionStep::K;
		scheme[5] = EncryptionStep::T;
		scheme[6] = EncryptionStep::E;
		scheme[7] = EncryptionStep::T;
		scheme[8] = EncryptionStep::T;
		scheme[9] = EncryptionStep::T;
		scheme[10] = EncryptionStep::K;
		scheme[11] = EncryptionStep::K;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::D;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::D;

		return scheme;
		};

	const auto check_image_equality = [](const BitmapImage& first, const BitmapImage& second) {
		const auto first_height = first.get_height();
		const auto second_height = second.get_height();
		const auto first_width = first.get_width();
		const auto second_width = second.get_width();

		if (first_height != second_height) {
			return false;
		}

		if (first_width != second_width) {
			return false;
		}

		try {
			for (auto y = 0; y < first_height; y++) {
				for (auto x = 0; x < first_width; x++) {
					const auto first_pixel = first.get_pixel(y, x);
					const auto second_pixel = second.get_pixel(y, x);

					if (first_pixel != second_pixel) {
						return false;
					}
				}
			}
		}
		catch (std::exception ex) {
			for (auto y = 0; y < first_height; y++) {
				for (auto x = 0; x < first_width; x++) {
					const auto first_pixel = first.get_pixel(x, y);
					const auto second_pixel = second.get_pixel(x, y);

					if (first_pixel != second_pixel) {
						return false;
					}
				}
			}
		}

		return true;
		};

	const auto scheme = get_scheme();
	const auto key = Key::get_standard_key();
	const auto image = get_image();

	const auto calculated_image = perform_scheme(image, key, scheme);

	const auto image_after_step_1 = FES::encrypt(image, key);
	const auto image_after_step_2 = FES::decrypt(image_after_step_1, key);
	const auto image_after_step_3 = FES::decrypt(image_after_step_2, key);
	const auto key_after_step_4 = Key::produce_new_key(key);
	const auto key_after_step_5 = Key::produce_new_key(key_after_step_4);
	const auto image_after_step_6 = image_after_step_3.transpose();
	const auto image_after_step_7 = FES::encrypt(image_after_step_6, key_after_step_5);
	const auto image_after_step_8 = image_after_step_7.transpose();
	const auto image_after_step_9 = image_after_step_8.transpose();
	const auto image_after_step_10 = image_after_step_9.transpose();
	const auto key_after_step_11 = Key::produce_new_key(key_after_step_5);
	const auto key_after_step_12 = Key::produce_new_key(key_after_step_11);
	const auto image_after_step_13 = FES::encrypt(image_after_step_10, key_after_step_12);
	const auto image_after_step_14 = FES::decrypt(image_after_step_13, key_after_step_12);
	const auto image_after_step_15 = FES::encrypt(image_after_step_14, key_after_step_12);
	const auto image_after_step_16 = FES::decrypt(image_after_step_15, key_after_step_12);

	const auto images_same = check_image_equality(calculated_image, image_after_step_16);

	ASSERT_TRUE(images_same);
}
