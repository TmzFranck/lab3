#include "test.h"
#include "test_declarations.h"

#include "cuda/common.cuh"
#include "cuda/encryption.cuh"
#include "encryption/Algorithm.h"
#include "util/Hash.h"

#include <cstdint>
#include <unordered_set>
#include <vector>

class HashTest : public LabTest {};

TEST_F(HashTest, test_three_a_one) {
	const auto test_kernel = [](const auto number_values, const auto number_threads, const auto number_blocks) {
		auto values = std::vector<std::uint64_t>(number_values);
		auto hashes_cpu = std::vector<std::uint64_t>(number_values);
		auto hashes_gpu = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto proto_value = static_cast<std::uint64_t>(i) + 137 * number_threads + number_blocks;
			const auto value = Hash::hash(proto_value);
			const auto hash = Hash::hash(value);

			values[i] = value;
			hashes_cpu[i] = hash;
		}

		auto valid = test_three_a_adapter(values.data(), hashes_gpu.data(), number_values, number_threads, number_blocks);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto cpu_value = hashes_cpu[i];
			const auto gpu_value = hashes_gpu[i];

			ASSERT_EQ(cpu_value, gpu_value);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 1024U;

	test_kernel(number_values_1, number_threads_1, divup(number_values_1, number_threads_1));

	const auto number_values_2 = 1024U;
	const auto number_threads_2 = 32U;

	test_kernel(number_values_2, number_threads_2, divup(number_values_2, number_threads_2));

	const auto number_values_3 = 2048U;
	const auto number_threads_3 = 64U;

	test_kernel(number_values_3, number_threads_3, divup(number_values_3, number_threads_3));
}

TEST_F(HashTest, test_three_a_two) {
	const auto test_kernel = [](const auto number_values, const auto number_threads, const auto number_blocks) {
		auto values = std::vector<std::uint64_t>(number_values);
		auto hashes_cpu = std::vector<std::uint64_t>(number_values);
		auto hashes_gpu = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto proto_value = static_cast<std::uint64_t>(i) + 137 * number_threads + number_blocks;
			const auto value = Hash::hash(proto_value);
			const auto hash = Hash::hash(value);

			values[i] = value;
			hashes_cpu[i] = hash;
		}

		auto valid = test_three_a_adapter(values.data(), hashes_gpu.data(), number_values, number_threads, number_blocks);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto cpu_value = hashes_cpu[i];
			const auto gpu_value = hashes_gpu[i];

			ASSERT_EQ(cpu_value, gpu_value);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 37U;

	test_kernel(number_values_1, number_threads_1, divup(number_values_1, number_threads_1));

	const auto number_values_2 = 1025U;
	const auto number_threads_2 = 32U;

	test_kernel(number_values_2, number_threads_2, divup(number_values_2, number_threads_2));

	const auto number_values_3 = 10000U;
	const auto number_threads_3 = 111U;

	test_kernel(number_values_3, number_threads_3, divup(number_values_3, number_threads_3));
}

TEST_F(HashTest, test_three_b_one) {
	const auto test_kernel = [](const auto number_values, const auto number_threads) {
		auto values = std::vector<std::uint64_t>(number_values);
		auto hashes_cpu = std::vector<std::uint64_t>(number_values);
		auto hashes_gpu = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto proto_value = static_cast<std::uint64_t>(i) + 137 * number_threads + (number_threads % 13);
			const auto value = Hash::hash(proto_value);
			const auto hash = Hash::hash(value);

			values[i] = value;
			hashes_cpu[i] = hash;
		}

		auto valid = test_three_b_adapter(values.data(), hashes_gpu.data(), number_values, number_threads);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto cpu_value = hashes_cpu[i];
			const auto gpu_value = hashes_gpu[i];

			ASSERT_EQ(cpu_value, gpu_value);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 1024U;

	test_kernel(number_values_1, number_threads_1);

	const auto number_values_2 = 512U;
	const auto number_threads_2 = 1024U;

	test_kernel(number_values_2, number_threads_2);

	const auto number_values_3 = 2048U;
	const auto number_threads_3 = 64U;

	test_kernel(number_values_3, number_threads_3);
}

TEST_F(HashTest, test_three_b_two) {
	const auto test_kernel = [](const auto number_values, const auto number_threads) {
		auto values = std::vector<std::uint64_t>(number_values);
		auto hashes_cpu = std::vector<std::uint64_t>(number_values);
		auto hashes_gpu = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto proto_value = static_cast<std::uint64_t>(i) + 137 * number_threads + (number_threads % 13);
			const auto value = Hash::hash(proto_value);
			const auto hash = Hash::hash(value);

			values[i] = value;
			hashes_cpu[i] = hash;
		}

		auto valid = test_three_b_adapter(values.data(), hashes_gpu.data(), number_values, number_threads);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto cpu_value = hashes_cpu[i];
			const auto gpu_value = hashes_gpu[i];

			ASSERT_EQ(cpu_value, gpu_value);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 37U;

	test_kernel(number_values_1, number_threads_1);

	const auto number_values_2 = 1025U;
	const auto number_threads_2 = 32U;

	test_kernel(number_values_2, number_threads_2);

	const auto number_values_3 = 10000U;
	const auto number_threads_3 = 111U;

	test_kernel(number_values_3, number_threads_3);
}

TEST_F(HashTest, test_three_c_one) {
	const auto get_no_hits = [](const auto length, const std::uint64_t value) {
		auto vec = std::vector<std::uint64_t>(length);

		for (auto i = 0; i < length; i++) {
			auto val = Hash::hash(static_cast<std::uint64_t>(i));
			if (val == value) {
				val++;
			}

			vec[i] = val;
		}

		return vec;
		};

	const auto get_all_hits = [](const auto length, const std::uint64_t value) {
		auto vec = std::vector<std::uint64_t>(length);

		for (auto i = 0; i < length; i++) {
			vec[i] = value;
		}

		return vec;
		};

	const auto get_few_hits = [](const auto length, const std::uint64_t value, const auto number_hits) {
		auto vec = std::vector<std::uint64_t>(length);

		for (auto i = 0; i < length; i++) {
			auto val = Hash::hash(static_cast<std::uint64_t>(i));
			if (val == value) {
				val++;
			}

			vec[i] = val;
		}

		for (auto step = 13; step < length; step += (length / number_hits)) {
			vec[step] = value;
		}

		return vec;
		};

	const auto test_kernel = [](const std::uint64_t value, const std::vector<std::uint64_t>& vec, const auto number_threads, const auto number_blocks) {
		const auto length = vec.size();
		auto indices = std::vector<unsigned int>(length);

		auto expected_indices = std::unordered_set<unsigned int>{};
		for (auto i = 0; i < length; i++) {
			indices[i] = 0;
			if (vec[i] == value) {
				expected_indices.emplace(i);
			}
		}

		auto valid = test_three_c_adapter(vec.data(), indices.data(), length, value, number_threads, number_blocks);
		ASSERT_TRUE(valid);

		auto found_zero_index = false;

		auto found_indices = std::unordered_set<unsigned int>{};
		for (auto i = 0; i < length; i++) {
			if (indices[i] == 0 && found_zero_index) {
				break;
			}

			if (indices[i] == 0) {
				found_zero_index = true;

				if (vec[indices[i]] == value) {
					found_indices.emplace(indices[i]);
				}

				continue;
			}

			found_indices.emplace(indices[i]);
		}

		ASSERT_EQ(expected_indices.size(), found_indices.size());
		for (const auto expected : expected_indices) {
			ASSERT_TRUE(found_indices.contains(expected));
		}
		};

	test_kernel(1374546, get_no_hits(1024, 1374546), 1024, 1);
	test_kernel(9847361, get_no_hits(1024, 9847361), 32, 32);
	test_kernel(88884444, get_no_hits(512, 88884444), 1024, 1);

	test_kernel(1374546, get_all_hits(1024, 1374546), 1024, 1);
	test_kernel(9847361, get_all_hits(1024, 9847361), 32, 32);
	test_kernel(88884444, get_all_hits(512, 88884444), 1024, 1);

	test_kernel(1374546, get_few_hits(1024, 1374546, 8), 1024, 1);
	test_kernel(9847361, get_few_hits(1024, 9847361, 8), 32, 32);
	test_kernel(88884444, get_few_hits(512, 88884444, 8), 1024, 1);
}

TEST_F(HashTest, test_three_c_two) {
	const auto get_no_hits = [](const auto length, const std::uint64_t value) {
		auto vec = std::vector<std::uint64_t>(length);

		for (auto i = 0; i < length; i++) {
			auto val = Hash::hash(static_cast<std::uint64_t>(i));
			if (val == value) {
				val++;
			}

			vec[i] = val;
		}

		return vec;
		};

	const auto get_all_hits = [](const auto length, const std::uint64_t value) {
		auto vec = std::vector<std::uint64_t>(length);

		for (auto i = 0; i < length; i++) {
			vec[i] = value;
		}

		return vec;
		};

	const auto get_few_hits = [](const auto length, const std::uint64_t value, const auto number_hits) {
		auto vec = std::vector<std::uint64_t>(length);

		for (auto i = 0; i < length; i++) {
			auto val = Hash::hash(static_cast<std::uint64_t>(i));
			if (val == value) {
				val++;
			}

			vec[i] = val;
		}

		for (auto step = 13; step < length; step += (length / number_hits)) {
			vec[step] = value;
		}

		return vec;
		};

	const auto test_kernel = [](const std::uint64_t value, const std::vector<std::uint64_t>& vec, const auto number_threads, const auto number_blocks) {
		const auto length = vec.size();
		auto indices = std::vector<unsigned int>(length);

		auto expected_indices = std::unordered_set<unsigned int>{};
		for (auto i = 0; i < length; i++) {
			indices[i] = 0;
			if (vec[i] == value) {
				expected_indices.emplace(i);
			}
		}

		auto valid = test_three_c_adapter(vec.data(), indices.data(), length, value, number_threads, number_blocks);
		ASSERT_TRUE(valid);

		auto found_zero_index = false;

		auto found_indices = std::unordered_set<unsigned int>{};
		for (auto i = 0; i < length; i++) {
			if (indices[i] == 0 && found_zero_index) {
				break;
			}

			if (indices[i] == 0) {
				found_zero_index = true;

				if (vec[indices[i]] == value) {
					found_indices.emplace(indices[i]);
				}

				continue;
			}

			found_indices.emplace(indices[i]);
		}

		ASSERT_EQ(expected_indices.size(), found_indices.size());
		for (const auto expected : expected_indices) {
			ASSERT_TRUE(found_indices.contains(expected));
		}
		};

	test_kernel(1374546, get_no_hits(1025U, 1374546), 32U, 33U);
	test_kernel(9847361, get_no_hits(10000U, 9847361), 111U, 100U);
	test_kernel(88884444, get_no_hits(1024U, 88884444), 37U, 40U);

	test_kernel(1374546, get_all_hits(1025U, 1374546), 32U, 33U);
	test_kernel(9847361, get_all_hits(10000U, 9847361), 111U, 100U);
	test_kernel(88884444, get_all_hits(1024U, 88884444), 37U, 40U);

	test_kernel(1374546, get_few_hits(1025U, 1374546, 54U), 32U, 33U);
	test_kernel(9847361, get_few_hits(10000U, 9847361, 54U), 111U, 100U);
	test_kernel(88884444, get_few_hits(1024U, 88884444, 54U), 37U, 40U);
}

TEST_F(HashTest, test_three_d_one) {
	const auto test_kernel = [](const auto number_values, const auto number_threads, const auto number_blocks) {
		auto hashes = std::vector<std::uint64_t>(number_values);
		auto expected_hashes = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto protoscheme = static_cast<std::uint64_t>(i);
			const auto scheme = (protoscheme & 4'294'967'295) | (protoscheme << 32);
			expected_hashes[i] = Hash::hash(scheme);
		}

		auto valid = test_three_d_adapter(hashes.data(), number_values, number_threads, number_blocks);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto expected_hash = expected_hashes[i];
			const auto computed_hash = hashes[i];
			ASSERT_EQ(expected_hash, computed_hash);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 1024U;
	const auto number_blocks_1 = 1U;

	test_kernel(number_values_1, number_threads_1, number_blocks_1);

	const auto number_values_2 = 512U;
	const auto number_threads_2 = 1024U;
	const auto number_blocks_2 = 2U;

	test_kernel(number_values_2, number_threads_2, number_blocks_2);

	const auto number_values_3 = 2048U;
	const auto number_threads_3 = 64U;
	const auto number_blocks_3 = 2U;

	test_kernel(number_values_3, number_threads_3, number_blocks_3);
}

TEST_F(HashTest, test_three_d_two) {
	const auto test_kernel = [](const auto number_values, const auto number_threads, const auto number_blocks) {
		auto hashes = std::vector<std::uint64_t>(number_values);
		auto expected_hashes = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto protoscheme = static_cast<std::uint64_t>(i);
			const auto scheme = (protoscheme & 4'294'967'295) | (protoscheme << 32);
			expected_hashes[i] = Hash::hash(scheme);
		}

		auto valid = test_three_d_adapter(hashes.data(), number_values, number_threads, number_blocks);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto expected_hash = expected_hashes[i];
			const auto computed_hash = hashes[i];
			ASSERT_EQ(expected_hash, computed_hash);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 37U;
	const auto number_blocks_1 = 1U;

	test_kernel(number_values_1, number_threads_1, number_blocks_1);

	const auto number_values_2 = 512U;
	const auto number_threads_2 = 231U;
	const auto number_blocks_2 = 14U;

	test_kernel(number_values_2, number_threads_2, number_blocks_2);

	const auto number_values_3 = 2048U;
	const auto number_threads_3 = 516U;
	const auto number_blocks_3 = 8U;

	test_kernel(number_values_3, number_threads_3, number_blocks_3);
}

TEST_F(HashTest, test_three_e_one) {
	const auto get_scheme_1 = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::D;
		scheme[1] = EncryptionStep::D;
		scheme[2] = EncryptionStep::D;
		scheme[3] = EncryptionStep::D;
		scheme[4] = EncryptionStep::D;
		scheme[5] = EncryptionStep::D;
		scheme[6] = EncryptionStep::D;
		scheme[7] = EncryptionStep::D;
		scheme[8] = EncryptionStep::D;
		scheme[9] = EncryptionStep::D;
		scheme[10] = EncryptionStep::E;
		scheme[11] = EncryptionStep::E;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::E;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::E;

		return scheme;
		};

	const auto get_scheme_2 = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::T;
		scheme[1] = EncryptionStep::T;
		scheme[2] = EncryptionStep::T;
		scheme[3] = EncryptionStep::T;
		scheme[4] = EncryptionStep::T;
		scheme[5] = EncryptionStep::T;
		scheme[6] = EncryptionStep::T;
		scheme[7] = EncryptionStep::T;
		scheme[8] = EncryptionStep::T;
		scheme[9] = EncryptionStep::T;
		scheme[10] = EncryptionStep::E;
		scheme[11] = EncryptionStep::E;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::E;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::E;

		return scheme;
		};

	const auto get_scheme_3 = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::E;
		scheme[1] = EncryptionStep::E;
		scheme[2] = EncryptionStep::E;
		scheme[3] = EncryptionStep::E;
		scheme[4] = EncryptionStep::E;
		scheme[5] = EncryptionStep::E;
		scheme[6] = EncryptionStep::E;
		scheme[7] = EncryptionStep::E;
		scheme[8] = EncryptionStep::E;
		scheme[9] = EncryptionStep::E;
		scheme[10] = EncryptionStep::E;
		scheme[11] = EncryptionStep::E;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::E;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::E;

		return scheme;
		};

	const auto get_scheme_4 = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::K;
		scheme[1] = EncryptionStep::K;
		scheme[2] = EncryptionStep::K;
		scheme[3] = EncryptionStep::K;
		scheme[4] = EncryptionStep::K;
		scheme[5] = EncryptionStep::K;
		scheme[6] = EncryptionStep::K;
		scheme[7] = EncryptionStep::K;
		scheme[8] = EncryptionStep::K;
		scheme[9] = EncryptionStep::K;
		scheme[10] = EncryptionStep::E;
		scheme[11] = EncryptionStep::E;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::E;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::E;

		return scheme;
		};

	const auto test = [](const auto& scheme) {
		const auto code = encode(scheme);
		const auto hash = Hash::hash(code);
		const auto retrieved_scheme = retrieve_scheme(hash);

		ASSERT_EQ(Hash::hash(encode(retrieved_scheme)), hash);
		};

	test(get_scheme_1());
	test(get_scheme_2());
	test(get_scheme_3());
	test(get_scheme_4());
}

TEST_F(HashTest, test_three_e_two) {
	const auto get_scheme_1 = []() {
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

		return scheme;
		};

	const auto get_scheme_2 = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::E;
		scheme[1] = EncryptionStep::D;
		scheme[2] = EncryptionStep::E;
		scheme[3] = EncryptionStep::D;
		scheme[4] = EncryptionStep::E;
		scheme[5] = EncryptionStep::D;
		scheme[6] = EncryptionStep::E;
		scheme[7] = EncryptionStep::D;
		scheme[8] = EncryptionStep::E;
		scheme[9] = EncryptionStep::D;
		scheme[10] = EncryptionStep::E;
		scheme[11] = EncryptionStep::E;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::E;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::E;

		return scheme;
		};

	const auto get_scheme_3 = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::T;
		scheme[1] = EncryptionStep::K;
		scheme[2] = EncryptionStep::T;
		scheme[3] = EncryptionStep::K;
		scheme[4] = EncryptionStep::T;
		scheme[5] = EncryptionStep::K;
		scheme[6] = EncryptionStep::T;
		scheme[7] = EncryptionStep::K;
		scheme[8] = EncryptionStep::T;
		scheme[9] = EncryptionStep::K;
		scheme[10] = EncryptionStep::E;
		scheme[11] = EncryptionStep::E;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::E;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::E;

		return scheme;
		};

	const auto get_scheme_4 = []() {
		auto scheme = EncryptionScheme{};
		scheme[0] = EncryptionStep::E;
		scheme[1] = EncryptionStep::T;
		scheme[2] = EncryptionStep::D;
		scheme[3] = EncryptionStep::K;
		scheme[4] = EncryptionStep::E;
		scheme[5] = EncryptionStep::T;
		scheme[6] = EncryptionStep::D;
		scheme[7] = EncryptionStep::K;
		scheme[8] = EncryptionStep::E;
		scheme[9] = EncryptionStep::T;
		scheme[10] = EncryptionStep::E;
		scheme[11] = EncryptionStep::E;
		scheme[12] = EncryptionStep::E;
		scheme[13] = EncryptionStep::E;
		scheme[14] = EncryptionStep::E;
		scheme[15] = EncryptionStep::E;

		return scheme;
		};

	const auto test = [](const auto& scheme) {
		const auto code = encode(scheme);
		const auto hash = Hash::hash(code);
		const auto retrieved_scheme = retrieve_scheme(hash);

		ASSERT_EQ(Hash::hash(encode(retrieved_scheme)), hash);
		};

	test(get_scheme_1());
	test(get_scheme_2());
	test(get_scheme_3());
	test(get_scheme_4());
}
