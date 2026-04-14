#pragma once
#include <cstddef>

struct AttnShape {
    int B;
    int H;
    int S;
    int D;

    size_t q_elems() const { return static_cast<size_t>(B) * H * D; }
    size_t k_elems() const { return static_cast<size_t>(B) * H * S * D; }
    size_t v_elems() const { return k_elems(); }
    size_t o_elems() const { return q_elems(); }
};
