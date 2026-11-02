#!/usr/bin/env python3
"""Danbooru tags for prompts of tag-based models, from a tag list (name,category,posts,"aliases").

  tags.py search WORDS... [--all] [-n N]   tags whose name or an alias contains the words, most used first
  tags.py check "PROMPT"                    each comma-separated tag of a prompt: known (posts), an alias (the
                                            name to use), rare, or unknown (the closest known tags)

Names print with spaces, as prompts write them. search leaves out artist and meta tags unless --all.
MAKO_TAGS_CSV: the list (default ~/models/tags/danbooru.csv).
"""
import csv
import itertools
import os
import re
import sys

CATEGORIES = {0: "general", 1: "artist", 3: "series", 4: "character", 5: "meta"}
RARE = 500  # posts below which a tag is likely barely learned
# Words prompts use by convention that are not Danbooru tags: quality, aesthetic and rating tags.
CONVENTIONS = {
    "masterpiece", "best quality", "amazing quality", "high quality", "good quality", "normal quality",
    "low quality", "worst quality", "bad quality", "worst detail", "very aesthetic", "aesthetic",
    "general", "sensitive", "nsfw", "explicit", "questionable", "safe", "newest", "recent", "mid", "early", "old",
}


def load(path):
    names, aliases = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                cat, posts = int(row[1]), int(row[2])
            except ValueError:
                continue
            names[row[0]] = (cat, posts)
            for a in (row[3].split(",") if len(row) > 3 and row[3] else []):
                a = a.strip()
                if a and not a.startswith("/"):
                    aliases.setdefault(a, row[0])
    return names, aliases


def key(text):
    """A prompt's tag as the list spells it: lowercase, underscores, no escapes."""
    return re.sub(r"\s+", "_", text.strip().lower().replace("\\", ""))


def show(name, names):
    cat, posts = names[name]
    return f"{name.replace('_', ' ')} ({CATEGORIES.get(cat, cat)}, {posts:,})"


def search(words, names, aliases, all_cats, n):
    q = key(" ".join(words))
    hits = {}
    for name in names:
        if q in name:
            hits[name] = None
    for a, name in aliases.items():
        if q in a and name not in hits:
            hits[name] = a
    ranked = sorted((h for h in hits if all_cats or names[h][0] not in (1, 5)),
                    key=lambda h: (h != q, q not in h.split("_") and not re.search(rf"(^|_){re.escape(q)}(_|$)", h),
                                   -names[h][1]))
    for h in ranked[:n]:
        print(show(h, names) + (f"  (alias {hits[h].replace('_', ' ')})" if hits[h] else ""))
    if not ranked:
        print(f"no tag contains {q.replace('_', ' ')!r}")


def tags_of(prompt):
    """The prompt's comma-separated tags, with weights and parentheses of emphasis taken off."""
    text = re.sub(r":\s*-?\d+(\.\d+)?\s*\)", ")", prompt)  # (tag:1.2)
    text = re.sub(r"(?<!\\)[()]", ",", text)  # emphasis parentheses split like commas
    return [t.strip() for t in text.split(",") if t.strip()]


def parts(t, names, aliases):
    """Known tags made of some of t's words in order ("long silver hair": long hair, silver hair), longest first."""
    words = t.split("_")
    found = []
    for k in range(len(words) - 1, 0, -1):
        for sub in itertools.combinations(words, k):
            s = "_".join(sub)
            name = s if s in names else aliases.get(s)
            if name and names[name][0] not in (1, 5) and name not in found and (k > 1 or names[name][1] >= RARE):
                found.append(name)
    return found[:5]


def close(t, names):
    """Known tags sharing the most of t's words (whole words), then the most used."""
    words = [w for w in re.split(r"[_()]+", t) if len(w) > 1]
    scored = []
    for n in names:
        if names[n][0] in (1, 5):
            continue
        nw = set(re.split(r"[_()]+", n))
        shared = sum(w in nw for w in words)
        if shared:
            scored.append((-shared, -names[n][1], n))
    return [n for _, _, n in sorted(scored)[:5]]


def check(prompt, names, aliases):
    for raw in tags_of(prompt):
        t = key(raw)
        shown = raw.replace("\\", "")
        if shown.lower() in CONVENTIONS:
            print(f"ok      {shown}  (a convention, not a Danbooru tag)")
        elif len(t.split("_")) > 5:
            print(f"prose   {shown}  (not a tag: split it into tags)")
        elif t in names:
            posts = names[t][1]
            print(f"{'rare' if posts < RARE else 'ok':7} {show(t, names)}")
        elif t in aliases:
            print(f"alias   {shown}  -> use {show(aliases[t], names)}")
        else:
            near = parts(t, names, aliases) or close(t, names)
            print(f"unknown {shown}" + (f"  -> maybe: {'; '.join(show(n, names) for n in near)}" if near else ""))


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("search", "check"):
        sys.exit(__doc__)
    path = os.environ.get("MAKO_TAGS_CSV", os.path.expanduser("~/models/tags/danbooru.csv"))
    if not os.path.exists(path):
        sys.exit(f"no tag list at {path} (MAKO_TAGS_CSV)")
    names, aliases = load(path)
    if args[0] == "search":
        all_cats = "--all" in args
        n = 20
        rest = [a for a in args[1:] if a != "--all"]
        if "-n" in rest:
            i = rest.index("-n")
            n = int(rest[i + 1])
            del rest[i:i + 2]
        if not rest:
            sys.exit("search: which words?")
        search(rest, names, aliases, all_cats, n)
    else:
        if len(args) != 2:
            sys.exit('check: one prompt, in quotes')
        check(args[1], names, aliases)


if __name__ == "__main__":
    main()
