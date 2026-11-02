#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mako::image {

// CLIP's byte-level BPE (vocab.json and merges.txt): text in NFC, runs of whitespace as one space, lowercased; split
// into contractions, letter runs, single digits and runs of other non-space characters; each piece's last symbol
// marked as a word's end ("</w>"). "<|startoftext|>" and "<|endoftext|>" in the text are their tokens.
class ClipTokenizer {
public:
    static std::expected<ClipTokenizer, std::string> load(const std::string& vocab_json, const std::string& merges_txt);

    // The text's tokens, without the start and end tokens.
    std::vector<int32_t> encode(std::string_view text) const;
    static constexpr int32_t kStart = 49406, kEnd = 49407;

    // Exposed for tests: the pieces of normalized text.
    static std::vector<std::string> pieces(std::string_view text);
    static std::string normalize(std::string_view text);

private:
    void bpe(const std::string& piece, std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> rank_; // "left right" -> merge rank
};

} // namespace mako::image
