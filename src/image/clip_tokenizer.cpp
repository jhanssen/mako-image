#include "image/clip_tokenizer.hpp"

#include <CoreFoundation/CoreFoundation.h>
#include <glaze/glaze.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <limits>

namespace mako::image {

namespace {

struct CodepointRange {
    uint32_t first, last;
};

#include "text/unicode_tables.inc"

template <size_t N>
bool in_ranges(const CodepointRange (&r)[N], uint32_t cp)
{
    auto it = std::upper_bound(std::begin(r), std::end(r), cp,
                               [](uint32_t c, const CodepointRange& x) { return c < x.first; });
    return it != std::begin(r) && cp <= std::prev(it)->last;
}

bool is_letter(uint32_t cp) { return in_ranges(kLetterRanges, cp); }
bool is_number(uint32_t cp) { return in_ranges(kNumberRanges, cp); }

// The Unicode White_Space property.
bool is_space(uint32_t cp)
{
    switch (cp) {
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20: case 0x85: case 0xA0: case 0x1680:
    case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
        return true;
    default:
        return cp >= 0x2000 && cp <= 0x200A;
    }
}

// Code points of UTF-8 text (an invalid byte is U+FFFD).
std::vector<uint32_t> decode(std::string_view s)
{
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const auto b = uint8_t(s[i]);
        const int n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
        uint32_t cp = n == 1 ? b : n == 2 ? b & 0x1f : n == 3 ? b & 0x0f : b & 0x07;
        bool ok = n > 0 && i + size_t(n) <= s.size();
        for (int k = 1; ok && k < n; ++k) {
            const auto c = uint8_t(s[i + size_t(k)]);
            ok = (c >> 6) == 2;
            cp = cp << 6 | (c & 0x3f);
        }
        out.push_back(ok ? cp : 0xFFFD);
        i += ok ? size_t(n) : 1;
    }
    return out;
}

void append_utf8(std::string& out, uint32_t cp)
{
    if (cp < 0x80) {
        out += char(cp);
    } else if (cp < 0x800) {
        out += char(0xC0 | (cp >> 6));
        out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += char(0xE0 | (cp >> 12));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    } else {
        out += char(0xF0 | (cp >> 18));
        out += char(0x80 | ((cp >> 12) & 0x3F));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    }
}

// The byte-level alphabet: printable bytes are themselves, the rest U+0100 and up.
const std::array<std::string, 256>& byte_text()
{
    static const auto table = [] {
        std::array<std::string, 256> t;
        uint32_t next = 256;
        for (uint32_t b = 0; b < 256; ++b) {
            const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
            append_utf8(t[b], printable ? b : next++);
        }
        return t;
    }();
    return table;
}

// NFC, then lowercase (Unicode's default mappings, as CoreFoundation applies them without a locale).
std::string nfc_lower(std::string_view s, bool lower)
{
    CFStringRef in = CFStringCreateWithBytes(kCFAllocatorDefault, reinterpret_cast<const UInt8*>(s.data()),
                                             CFIndex(s.size()), kCFStringEncodingUTF8, false);
    if (!in) { // not UTF-8: ASCII rules only
        std::string out(s);
        if (lower)
            for (char& c : out)
                c = char(std::tolower(static_cast<unsigned char>(c)));
        return out;
    }
    CFMutableStringRef m = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, in);
    CFRelease(in);
    CFStringNormalize(m, kCFStringNormalizationFormC);
    if (lower)
        CFStringLowercase(m, nullptr);
    const CFIndex len = CFStringGetLength(m);
    CFIndex bytes = 0;
    CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false, nullptr, 0, &bytes);
    std::string out(size_t(bytes), '\0');
    CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false, reinterpret_cast<UInt8*>(out.data()),
                     bytes, nullptr);
    CFRelease(m);
    return out;
}

std::string read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

std::expected<ClipTokenizer, std::string> ClipTokenizer::load(const std::string& vocab_json,
                                                              const std::string& merges_txt)
{
    ClipTokenizer t;
    const std::string vtext = read_file(vocab_json);
    glz::generic v;
    if (vtext.empty() || glz::read_json(v, vtext) || !v.is_object())
        return std::unexpected(vocab_json + ": not a JSON object");
    for (const auto& [k, id] : v.get_object()) {
        if (!id.is_number())
            return std::unexpected(vocab_json + ": a token id that is not a number");
        t.vocab_[k] = int32_t(id.get_number());
    }
    std::ifstream in(merges_txt);
    if (!in)
        return std::unexpected("cannot open " + merges_txt);
    std::string line;
    // The first line names the format; CLIP uses the next 48894 merges (49152 tokens less the bytes, their word-end
    // forms and the two special tokens).
    constexpr int32_t kMerges = 49152 - 256 - 2;
    std::getline(in, line);
    for (int32_t r = 0; r < kMerges && std::getline(in, line); ++r) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        t.rank_.emplace(line, r);
    }
    if (t.vocab_.size() < 49408 || t.rank_.empty())
        return std::unexpected(vocab_json + ": not CLIP's vocabulary");
    return t;
}

std::string ClipTokenizer::normalize(std::string_view text)
{
    const auto cps = decode(nfc_lower(text, false));
    std::string spaced;
    for (size_t i = 0; i < cps.size(); ++i) {
        // Capital sigma lowercases to the plain form, also where it ends a word (CoreFoundation's would be final).
        if (cps[i] == 0x03A3) {
            append_utf8(spaced, 0x03C3);
        } else if (is_space(cps[i])) {
            while (i + 1 < cps.size() && is_space(cps[i + 1]))
                ++i;
            spaced += ' ';
        } else {
            append_utf8(spaced, cps[i]);
        }
    }
    return nfc_lower(spaced, true);
}

std::vector<std::string> ClipTokenizer::pieces(std::string_view text)
{
    // 's|'t|'re|'ve|'m|'ll|'d | \pL+ | \pN | [^\s\pL\pN]+, leftmost alternative first; what none matches (spaces) is
    // dropped.
    const auto c = decode(text);
    std::vector<std::string> out;
    auto other = [&](uint32_t cp) { return !is_space(cp) && !is_letter(cp) && !is_number(cp); };
    for (size_t i = 0; i < c.size();) {
        size_t j = i;
        if (c[i] == '\'' && i + 1 < c.size()) {
            const uint32_t a = c[i + 1], b = i + 2 < c.size() ? c[i + 2] : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd')
                j = i + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
                j = i + 3;
        }
        if (j == i && is_letter(c[i]))
            while (j < c.size() && is_letter(c[j]))
                ++j;
        if (j == i && is_number(c[i]))
            j = i + 1;
        if (j == i && other(c[i]))
            while (j < c.size() && other(c[j]))
                ++j;
        if (j == i) { // a space
            ++i;
            continue;
        }
        std::string p;
        for (size_t k = i; k < j; ++k)
            append_utf8(p, c[k]);
        out.push_back(std::move(p));
        i = j;
    }
    return out;
}

void ClipTokenizer::bpe(const std::string& piece, std::vector<int32_t>& out) const
{
    const auto& bt = byte_text();
    std::vector<std::string> sym;
    for (unsigned char b : piece)
        sym.push_back(bt[b]);
    if (sym.empty())
        return;
    sym.back() += "</w>";
    while (sym.size() > 1) {
        int32_t best = std::numeric_limits<int32_t>::max();
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i)
            if (const auto it = rank_.find(sym[i] + " " + sym[i + 1]); it != rank_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        if (best == std::numeric_limits<int32_t>::max())
            break;
        // Every occurrence of the best pair, left to right.
        const std::string a = sym[at], b = sym[at + 1];
        std::vector<std::string> next;
        for (size_t i = 0; i < sym.size(); ++i)
            if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) {
                next.push_back(a + b);
                ++i;
            } else {
                next.push_back(sym[i]);
            }
        sym = std::move(next);
    }
    for (const auto& s : sym)
        if (const auto it = vocab_.find(s); it != vocab_.end())
            out.push_back(it->second);
}

std::vector<int32_t> ClipTokenizer::encode(std::string_view text) const
{
    std::vector<int32_t> out;
    static constexpr std::pair<std::string_view, int32_t> kSpecial[] = {{"<|startoftext|>", kStart},
                                                                         {"<|endoftext|>", kEnd}};
    while (!text.empty()) {
        size_t at = std::string_view::npos;
        int32_t id = 0;
        size_t len = 0;
        for (const auto& [s, i] : kSpecial)
            if (const size_t p = text.find(s); p < at) {
                at = p;
                id = i;
                len = s.size();
            }
        for (const auto& p : pieces(normalize(text.substr(0, at))))
            bpe(p, out);
        if (at == std::string_view::npos)
            break;
        out.push_back(id);
        text.remove_prefix(at + len);
    }
    return out;
}

} // namespace mako::image
