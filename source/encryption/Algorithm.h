#include <array>
#include <cstdint>
#include <exception>
#include "image/bitmap_image.h"
#include "Key.h"
#include "FES.h"
#include "util/Hash.h"
#pragma once

enum EncryptionStep { E, D, K, T };
using EncryptionScheme = std::array<EncryptionStep, 16>;

std::uint64_t encode(const EncryptionScheme& scheme) {
    std::uint64_t result = 0;

    for (int i = 0; i < 16; ++i) {
        switch (scheme[i]) {
            case E:
                result |= (0b00 << (2 * i));
                break;
            case D:
                result |= (0b01 << (2 * i));
                break;
            case K:
                result |= (0b10 << (2 * i));
                break;
            case T:
                result |= (0b11 << (2 * i));
                break;
        }
    }

    // Replizieren Sie die 32 unteren Bits in die 32 oberen Bits
    result |= (result << 32);

    return result;
}


EncryptionScheme decode(std::uint64_t code) {
    // Überprüfen, ob die unteren 32 Bit den oberen 32 Bit entsprechen
    if ((code & 0xFFFFFFFF) != ((code >> 32) & 0xFFFFFFFF)) {
        throw std::exception();
    }

    EncryptionScheme scheme;

    for (int i = 0; i < 16; ++i) {
        std::uint64_t bits = (code >> (2 * i)) & 0b11;

        switch (bits) {
            case 0b00:
                scheme[i] = E;
                break;
            case 0b01:
                scheme[i] = D;
                break;
            case 0b10:
                scheme[i] = K;
                break;
            case 0b11:
                scheme[i] = T;
                break;
        }
    }

    return scheme;
}

BitmapImage perform_scheme(BitmapImage image, Key::key_type key, const EncryptionScheme& scheme) {
    for (EncryptionStep step : scheme) {
        switch (step) {
            case E:
                image = FES::encrypt(image, key);
                break;
            case D:
                image = FES::decrypt(image, key);
                break;
            case T:
                image = image.transpose();
                break;
            case K:
                key = Key::produce_new_key(key);
                break;
        }
    }

    return image;
}

EncryptionScheme retrieve_scheme(std::uint64_t hash) {
    // Erstellen Sie ein leeres Schema
    EncryptionScheme scheme;

    // Füllen Sie die letzten 6 Schritte des Schemas mit E
    for (int i = 10; i < 16; ++i) {
        scheme[i] = E;
    }

    // Dekodieren Sie die ersten 10 Schritte des Schemas aus dem Hash
    for (int i = 0; i < 10; ++i) {
        std::uint64_t bits = (hash >> (2 * i)) & 0b11;

        switch (bits) {
            case 0b00:
                scheme[i] = E;
                break;
            case 0b01:
                scheme[i] = D;
                break;
            case 0b10:
                scheme[i] = K;
                break;
            case 0b11:
                scheme[i] = T;
                break;
        }
    }

    // Überprüfen Sie, ob der Hash des Schemas dem gegebenen Hash entspricht
    if (hash != Hash::hash(encode(scheme))) {
        throw std::exception();
    }

    return scheme;
}

