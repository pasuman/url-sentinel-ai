"""
Test script comparing original vs improved inference on false positive URLs.

Usage:
    python test_improvements.py
"""

import sys

from inference import classify_url as classify_original
from inference_improved import classify_url_improved

# False positive URLs reported by user
TEST_URLS = [
    ("Shinhan Bank", "https://me.shinhan.com/vbojae5x"),
    ("Immunife", "https://immunife.co.kr"),
    ("Seoul Gov", "https://ecomileage.seoul.go.kr/green.oath.do"),
    (
        "CJ Logistics",
        "https://dxsmapp.cjlogistics.com/mmspush/trust.do?empnum=NTkzNDMy&trspbillnum=Njk2MjM3MTE1NDUy",
    ),
    ("CJ News", "https://www.cjlogistics.com/ko/newsroom/latest/LT_00000028"),
    (
        "Meritz Fire",
        "https://mstore.meritzfire.com/TIS2510CA000044/southeaster.do",
    ),
    ("Vo.la 1", "https://vo.la/j6xmz3B"),
    ("Vo.la 2", "https://vo.la/mv2eHVl"),
    (
        "KB Card",
        "https://m.kbcard.com/CMN/DVIEW/AODMCXHDACOCD0005?cless=2575695666051211",
    ),
    ("US Campus", "https://us-campus.co.kr/products/academy260_3"),
    ("Emart", "https://chatbot.emart.com/flyers"),
    (
        "Woori Bank",
        "https://nwon.wooribank.com/sml/apps/mmh/HMN/NWMMH00015/NWMMH00015_001M?withyou=NWMMH00051_001M&PLM_PDCD=P010002585",
    ),
]

# Known phishing URLs for comparison
PHISHING_URLS = [
    ("PayPal Fake", "http://paypa1-secure.com/login.php"),
    ("Google Fake", "https://google-verify.tk/reset"),
    ("Apple Fake", "https://apple-security.ml/account/verify"),
]


def main():
    print("=" * 90)
    print("COMPARING ORIGINAL vs IMPROVED INFERENCE")
    print("=" * 90)

    print("\n📋 LEGITIMATE KOREAN URLS (Should be classified as LEGIT)")
    print("-" * 90)

    legit_fp_original = 0
    legit_fp_improved = 0

    for name, url in TEST_URLS:
        prob_orig = classify_original(url)
        result_improved = classify_url_improved(url)

        is_fp_orig = prob_orig > 0.5
        is_fp_improved = result_improved["is_phishing"]

        if is_fp_orig:
            legit_fp_original += 1
        if is_fp_improved:
            legit_fp_improved += 1

        # Status indicators
        status_orig = "✗ FP" if is_fp_orig else "✓ OK"
        status_improved = "✗ FP" if is_fp_improved else "✓ OK"
        whitelist_tag = " [WHITELISTED]" if result_improved["whitelist_match"] else ""

        print(f"\n{name}:")
        print(f"  Original:  {prob_orig:.4f} {status_orig}")
        print(
            f"  Improved:  {result_improved['probability']:.4f} {status_improved}{whitelist_tag}"
        )

    print("\n" + "=" * 90)
    print("📊 LEGITIMATE URL RESULTS")
    print("-" * 90)
    print(f"Original:  {legit_fp_original}/{len(TEST_URLS)} false positives "
          f"({legit_fp_original/len(TEST_URLS)*100:.1f}%)")
    print(f"Improved:  {legit_fp_improved}/{len(TEST_URLS)} false positives "
          f"({legit_fp_improved/len(TEST_URLS)*100:.1f}%)")
    print(f"Reduction: {legit_fp_original - legit_fp_improved} false positives eliminated "
          f"(-{(legit_fp_original - legit_fp_improved)/len(TEST_URLS)*100:.1f}%)")

    print("\n" + "=" * 90)
    print("🎣 KNOWN PHISHING URLS (Should be classified as PHISHING)")
    print("-" * 90)

    phish_fn_original = 0
    phish_fn_improved = 0

    for name, url in PHISHING_URLS:
        prob_orig = classify_original(url)
        result_improved = classify_url_improved(url)

        is_fn_orig = prob_orig <= 0.5
        is_fn_improved = not result_improved["is_phishing"]

        if is_fn_orig:
            phish_fn_original += 1
        if is_fn_improved:
            phish_fn_improved += 1

        status_orig = "✗ FN" if is_fn_orig else "✓ OK"
        status_improved = "✗ FN" if is_fn_improved else "✓ OK"

        print(f"\n{name}:")
        print(f"  Original:  {prob_orig:.4f} {status_orig}")
        print(f"  Improved:  {result_improved['probability']:.4f} {status_improved}")

    print("\n" + "=" * 90)
    print("📊 PHISHING URL RESULTS")
    print("-" * 90)
    print(f"Original:  {phish_fn_original}/{len(PHISHING_URLS)} false negatives "
          f"({phish_fn_original/len(PHISHING_URLS)*100:.1f}%)")
    print(f"Improved:  {phish_fn_improved}/{len(PHISHING_URLS)} false negatives "
          f"({phish_fn_improved/len(PHISHING_URLS)*100:.1f}%)")

    print("\n" + "=" * 90)
    print("💡 SUMMARY")
    print("-" * 90)
    print(f"False Positive Reduction: {(1 - legit_fp_improved/legit_fp_original)*100:.1f}% "
          f"({legit_fp_original} → {legit_fp_improved})")
    if phish_fn_improved > phish_fn_original:
        print(f"⚠️  Trade-off: +{phish_fn_improved - phish_fn_original} false negatives "
              f"(precision ↑, recall ↓)")
    else:
        print("✓ No increase in false negatives")

    print("\n🔧 Improvements applied:")
    print("  • Weighted view voting (reduce URL/File, increase Domain/Network)")
    print("  • Korean domain whitelist (major banks, .kr TLDs)")
    print("  • Adjusted threshold (0.5 → 0.65)")
    print("  • Korean safe patterns (.do endpoints, .go.kr)")

    print("\n📝 Next steps for further improvement:")
    print("  1. Collect network features at runtime (DNS, SSL, WHOIS)")
    print("     → Expected: 60-70% FP reduction, AUC 0.58 → 0.92+")
    print("  2. Augment training data with 10K Korean legit URLs")
    print("     → Expected: 50%+ FP reduction on Korean domains")
    print("  3. Integrate Google Safe Browsing API for known phishing")
    print("     → Expected: High-confidence detection of known threats")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
