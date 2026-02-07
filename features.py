"""
URL feature extraction for phishing detection.

Parses a raw URL string into 97 numeric features matching the training data
column order (dataset_cybersecurity_michelle.csv with external columns dropped).

Usage:
    from features import extract_features, FEATURE_NAMES

    features = extract_features("https://example.com/path?q=1")
    # returns np.ndarray of shape (97,), dtype float32
"""

import ipaddress
import re
from urllib.parse import parse_qs, urlparse

import numpy as np

SPECIAL_CHARS = [".", "-", "_", "/", "?", "=", "@", "&", "!", " ", "~", ",", "+", "*", "#", "$", "%"]

CHAR_FEATURE_NAMES = [
    "qty_dot", "qty_hyphen", "qty_underline", "qty_slash",
    "qty_questionmark", "qty_equal", "qty_at", "qty_and",
    "qty_exclamation", "qty_space", "qty_tilde", "qty_comma",
    "qty_plus", "qty_asterisk", "qty_hashtag", "qty_dollar",
    "qty_percent",
]

COMPONENTS = ["url", "domain", "directory", "file", "params"]

# Build FEATURE_NAMES in the exact order of the training CSV (minus external cols)
FEATURE_NAMES: list[str] = []
# url: 17 char counts + qty_tld_url + length_url
FEATURE_NAMES.extend(f"{name}_url" for name in CHAR_FEATURE_NAMES)
FEATURE_NAMES.extend(["qty_tld_url", "length_url"])
# domain: 17 char counts + qty_vowels_domain + domain_length + domain_in_ip
FEATURE_NAMES.extend(f"{name}_domain" for name in CHAR_FEATURE_NAMES)
FEATURE_NAMES.extend(["qty_vowels_domain", "domain_length", "domain_in_ip"])
# directory: 17 char counts + directory_length
FEATURE_NAMES.extend(f"{name}_directory" for name in CHAR_FEATURE_NAMES)
FEATURE_NAMES.append("directory_length")
# file: 17 char counts + file_length
FEATURE_NAMES.extend(f"{name}_file" for name in CHAR_FEATURE_NAMES)
FEATURE_NAMES.append("file_length")
# params: 17 char counts + params_length + tld_present_params + qty_params
FEATURE_NAMES.extend(f"{name}_params" for name in CHAR_FEATURE_NAMES)
FEATURE_NAMES.extend(["params_length", "tld_present_params", "qty_params"])
# global features
FEATURE_NAMES.extend(["email_in_url", "url_shortened"])

assert len(FEATURE_NAMES) == 97, f"Expected 97 features, got {len(FEATURE_NAMES)}"

# Feature views for multi-view ensemble — maps view name to feature indices
FEATURE_VIEWS = {
    "url": list(range(0, 19)) + [95],       # 20: 17 chars + qty_tld_url + length_url + email_in_url
    "domain": list(range(19, 39)) + [96],    # 21: 17 chars + vowels + domain_length + domain_in_ip + url_shortened
    "directory": list(range(39, 57)),         # 18: 17 chars + directory_length
    "file": list(range(57, 75)),             # 18: 17 chars + file_length
    "params": list(range(75, 95)),           # 20: 17 chars + params_length + tld_present_params + qty_params
}

_VOWELS = set("aeiouAEIOU")

_EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.\w+")

_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "j.mp", "lnkd.in", "db.tt", "qr.ae",
    "cur.lv", "rebrand.ly", "rb.gy", "shorturl.at", "tiny.cc",
    "bl.ink", "short.io", "clck.ru", "v.gd", "tr.im", "soo.gd",
    "s2r.co", "cutt.ly", "shrtco.de",
}

# Common TLDs for qty_tld_url counting
_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "info", "biz", "name", "pro", "aero", "coop", "museum",
    "ac", "ad", "ae", "af", "ag", "ai", "al", "am", "an", "ao", "aq", "ar", "as", "at", "au", "aw", "ax", "az",
    "ba", "bb", "bd", "be", "bf", "bg", "bh", "bi", "bj", "bm", "bn", "bo", "br", "bs", "bt", "bv", "bw", "by", "bz",
    "ca", "cc", "cd", "cf", "cg", "ch", "ci", "ck", "cl", "cm", "cn", "co", "cr", "cu", "cv", "cw", "cx", "cy", "cz",
    "de", "dj", "dk", "dm", "do", "dz",
    "ec", "ee", "eg", "er", "es", "et", "eu",
    "fi", "fj", "fk", "fm", "fo", "fr",
    "ga", "gb", "gd", "ge", "gf", "gg", "gh", "gi", "gl", "gm", "gn", "gp", "gq", "gr", "gs", "gt", "gu", "gw", "gy",
    "hk", "hm", "hn", "hr", "ht", "hu",
    "id", "ie", "il", "im", "in", "io", "iq", "ir", "is", "it",
    "je", "jm", "jo", "jp",
    "ke", "kg", "kh", "ki", "km", "kn", "kp", "kr", "kw", "ky", "kz",
    "la", "lb", "lc", "li", "lk", "lr", "ls", "lt", "lu", "lv", "ly",
    "ma", "mc", "md", "me", "mg", "mh", "mk", "ml", "mm", "mn", "mo", "mp", "mq", "mr", "ms", "mt", "mu", "mv", "mw", "mx", "my", "mz",
    "na", "nc", "ne", "nf", "ng", "ni", "nl", "no", "np", "nr", "nu", "nz",
    "om",
    "pa", "pe", "pf", "pg", "ph", "pk", "pl", "pm", "pn", "pr", "ps", "pt", "pw", "py",
    "qa",
    "re", "ro", "rs", "ru", "rw",
    "sa", "sb", "sc", "sd", "se", "sg", "sh", "si", "sj", "sk", "sl", "sm", "sn", "so", "sr", "ss", "st", "su", "sv", "sx", "sy", "sz",
    "tc", "td", "tf", "tg", "th", "tj", "tk", "tl", "tm", "tn", "to", "tp", "tr", "tt", "tv", "tw", "tz",
    "ua", "ug", "uk", "us", "uy", "uz",
    "va", "vc", "ve", "vg", "vi", "vn", "vu",
    "wf", "ws",
    "ye", "yt",
    "za", "zm", "zw",
    # common generic TLDs
    "app", "dev", "xyz", "online", "site", "tech", "store", "shop", "club",
    "top", "wang", "win", "bid", "loan", "download", "racing", "date",
    "review", "science", "party", "stream", "trade", "faith", "accountant",
    "cricket", "work", "cloud", "live", "space", "website", "press",
    "host", "fun", "icu", "buzz", "mobi", "tel", "asia", "cat", "jobs",
    "travel", "xxx", "post",
}


def _count_chars(text: str) -> list[int]:
    """Count occurrences of each special character in text."""
    return [text.count(ch) for ch in SPECIAL_CHARS]


def _count_tlds(url: str) -> int:
    """Count number of known TLD strings appearing in the URL."""
    url_lower = url.lower()
    count = 0
    for tld in _TLDS:
        # Look for .tld pattern followed by end-of-string, /, :, ?, or another dot
        pattern = f".{tld}"
        start = 0
        while True:
            idx = url_lower.find(pattern, start)
            if idx == -1:
                break
            end = idx + len(pattern)
            if end == len(url_lower) or url_lower[end] in "/:?.&=#":
                count += 1
            start = end
    return count


def _is_ip_domain(domain: str) -> int:
    """Check if domain is an IP address."""
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0


def extract_features(url: str) -> np.ndarray:
    """
    Extract 97 URL-derivable features from a raw URL string.

    The features match the column order of the training CSV
    (dataset_cybersecurity_michelle.csv) after dropping the 14 external columns.

    Args:
        url: Raw URL string (e.g., "https://example.com/path?q=1").

    Returns:
        np.ndarray of shape (97,) with dtype float32.
    """
    features: list[float] = []

    # Ensure URL has a scheme for urlparse
    url_for_parse = url
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url_for_parse = "http://" + url

    parsed = urlparse(url_for_parse)

    # --- URL component (full URL string) ---
    features.extend(_count_chars(url))  # 17 char counts
    features.append(_count_tlds(url))   # qty_tld_url
    features.append(len(url))           # length_url

    # --- Domain ---
    domain = parsed.hostname or ""
    features.extend(_count_chars(domain))  # 17 char counts
    features.append(sum(1 for c in domain if c in _VOWELS))  # qty_vowels_domain
    features.append(len(domain))  # domain_length
    features.append(_is_ip_domain(domain) if domain else 0)  # domain_in_ip

    # --- Directory and File ---
    path = parsed.path or ""
    # Convention: directory = path up to and including last "/",
    # file = everything after last "/".
    # No path (empty or "/") → both directory and file are -1.
    if len(path) > 1:
        last_slash = path.rfind("/")
        # Directory: from start of path to last "/" (inclusive)
        directory = path[: last_slash + 1]
        features.extend(_count_chars(directory))
        features.append(len(directory))
        # File: after last "/"
        file_part = path[last_slash + 1:]
        features.extend(_count_chars(file_part))
        features.append(len(file_part))
    else:
        # No meaningful path → both directory and file are -1
        features.extend([-1] * 18)  # directory
        features.extend([-1] * 18)  # file

    # --- Params (query string) ---
    query = parsed.query
    if query:
        features.extend(_count_chars(query))
        features.append(len(query))
        # tld_present_params: check if any TLD appears in query
        query_lower = query.lower()
        tld_in_params = 0
        for tld in _TLDS:
            if f".{tld}" in query_lower:
                tld_in_params = 1
                break
        features.append(tld_in_params)
        # qty_params: number of query parameters
        features.append(len(parse_qs(query, keep_blank_values=True)))
    else:
        features.extend([-1] * 20)

    # --- Global features ---
    # email_in_url
    features.append(1 if _EMAIL_RE.search(url) else 0)

    # url_shortened
    domain_lower = (parsed.hostname or "").lower()
    # Also check with port stripped
    features.append(1 if domain_lower in _SHORTENERS else 0)

    result = np.array(features, dtype=np.float32)
    assert result.shape == (97,), f"Expected 97 features, got {result.shape[0]}"
    return result
