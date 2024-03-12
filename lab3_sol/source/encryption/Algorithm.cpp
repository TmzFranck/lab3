#include "Algorithm.h"

#include "cuda/encryption.cuh"

#include <cstdint>

EncryptionScheme retrieve_scheme(const std::uint64_t code) {
    const auto scheme = retrieve_scheme_adapter(code);
    const auto decoded_scheme = decode(scheme);

    return decoded_scheme;
}
