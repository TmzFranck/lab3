#include "cuda/image.cuh"

#include "encryption/Algorithm.h"
#include "encryption/FES.h"
#include "encryption/Key.h"
#include "io/image_parser.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <bit>

int main(int argc, char** argv) {
	auto lab_cli_app = CLI::App{ "" };

	auto image_path = std::filesystem::path{};
	auto file_option = lab_cli_app.add_option("--file", image_path);

	auto output_path = std::filesystem::path{};
	auto output_option = lab_cli_app.add_option("--output", output_path);

	file_option->check(CLI::ExistingFile);
	output_option->check(CLI::ExistingDirectory);

	CLI11_PARSE(lab_cli_app, argc, argv);

	auto loaded_image = ImageParser::read_bitmap(image_path);

	auto key = Key::get_standard_key();

	auto scheme = EncryptionScheme{};
	scheme[0] = EncryptionStep::E;
	scheme[1] = EncryptionStep::K;
	scheme[2] = EncryptionStep::K;
	scheme[3] = EncryptionStep::D;
	scheme[4] = EncryptionStep::T;
	scheme[5] = EncryptionStep::K;
	scheme[6] = EncryptionStep::E;
	scheme[7] = EncryptionStep::K;
	scheme[8] = EncryptionStep::D;
	scheme[9] = EncryptionStep::T;
	scheme[10] = EncryptionStep::E;
	scheme[11] = EncryptionStep::E;
	scheme[12] = EncryptionStep::E;
	scheme[13] = EncryptionStep::E;
	scheme[14] = EncryptionStep::E;
	scheme[15] = EncryptionStep::E;

	auto highly_encryption_image = perform_scheme(loaded_image, key, scheme);

	ImageParser::write_bitmap(output_path / "encrypted.bmp", highly_encryption_image);

	auto key_1 = Key::get_standard_key();
	auto key_2 = Key::produce_new_key(key_1);
	auto key_3 = Key::produce_new_key(key_2);
	auto key_4 = Key::produce_new_key(key_3);
	auto key_5 = Key::produce_new_key(key_4);

	auto img_16 = FES::decrypt(highly_encryption_image, key_5);
	auto img_15 = FES::decrypt(img_16, key_5);
	auto img_14 = FES::decrypt(img_15, key_5);
	auto img_13 = FES::decrypt(img_14, key_5);
	auto img_12 = FES::decrypt(img_13, key_5);
	auto img_11 = FES::decrypt(img_12, key_5);
	auto img_10 = img_11.transpose();
	auto img_09 = FES::encrypt(img_10, key_5);
	auto img_08 = img_09;
	auto img_07 = FES::decrypt(img_08, key_4);
	auto img_06 = img_07;
	auto img_05 = img_06.transpose();
	auto img_04 = FES::encrypt(img_05, key_3);
	auto img_03 = img_04;
	auto img_02 = img_03;
	auto img_01 = FES::decrypt(img_02, key_1);

	ImageParser::write_bitmap(output_path / "decrypted.bmp", img_01);

	return 0;
}
