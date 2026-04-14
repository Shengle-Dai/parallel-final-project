#pragma once
// Small helpers shared by main.cpp and benchmark.cpp for the --load-kv-from
// path: a minimal JSON int reader and a binary file slurp. These exist only
// so both binaries can consume the meta.json + *.bin layout that capture_kv.py
// (and tqkv_decode itself) produces.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

inline std::string tqkv_slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "could not open %s\n", path.c_str());
        std::exit(2);
    }
    std::ostringstream oss;
    oss << f.rdbuf();
    return oss.str();
}

inline int tqkv_read_meta_int(const std::string& contents, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    size_t k = contents.find(needle);
    if (k == std::string::npos) {
        std::fprintf(stderr, "meta.json: missing key \"%s\"\n", key);
        std::exit(2);
    }
    size_t colon = contents.find(':', k + needle.size());
    if (colon == std::string::npos) {
        std::fprintf(stderr, "meta.json: malformed near \"%s\"\n", key);
        std::exit(2);
    }
    size_t v = colon + 1;
    while (v < contents.size() && std::isspace(static_cast<unsigned char>(contents[v]))) ++v;
    int sign = 1;
    if (v < contents.size() && contents[v] == '-') { sign = -1; ++v; }
    if (v >= contents.size() || !std::isdigit(static_cast<unsigned char>(contents[v]))) {
        std::fprintf(stderr, "meta.json: non-integer value for \"%s\"\n", key);
        std::exit(2);
    }
    int n = 0;
    while (v < contents.size() && std::isdigit(static_cast<unsigned char>(contents[v]))) {
        n = n * 10 + (contents[v] - '0');
        ++v;
    }
    return sign * n;
}

inline void tqkv_read_bin(const std::string& path, void* p, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "could not open %s for reading\n", path.c_str());
        std::exit(2);
    }
    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(bytes));
    if (!f) {
        std::fprintf(stderr, "could not read %zu bytes from %s\n", bytes, path.c_str());
        std::exit(2);
    }
}
