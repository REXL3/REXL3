// The target sampler's keyed draw on the device, shared by every kernel that proposes with the TARGET's noise: the DFlash2 selector's walk
// (dflash2.cu) and the GLM NextN head's pick (glm_mtp.cu). It is `crates/sampling` (`uniform` + `try_choose`) in the same double arithmetic: splitmix64
// of (seed, position, token id) -> an open-interval uniform -> -ln(-ln u), added to logit / T over the candidates the sampler keeps (top_k, top_p, min_p).
// The noise of a candidate depends on its TOKEN ID alone (never on its column, its row or the launch), which is what lets a draft head that scores only
// a subset of the vocabulary draw the very noise the full head would.
//
// Moved verbatim out of dflash2.cu (a shared header, no other change): the Qwen goldens and the DFlash2 walk tests prove it moved no bit.
#pragma once
#include <math.h>

namespace gb_keyed {

// The target sampler's keyed Gumbel (crates/sampling `uniform` + `try_choose`), in the same double arithmetic: splitmix64 of
// (seed, position, token id) -> an open-interval uniform -> -ln(-ln u).
static __device__ __forceinline__ unsigned long long sm_mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static __device__ __forceinline__ double keyed_uniform(unsigned long long seed, unsigned long long position, unsigned long long id) {
    unsigned long long x = sm_mix(seed + 0x9E3779B97F4A7C15ULL);
    x = sm_mix(x ^ (position * 0xD1B54A32D192ED03ULL));
    x = sm_mix(x ^ id);
    const double two_m53 = 1.0 / 9007199254740992.0;
    const double u = (double)(x >> 11) * two_m53 + two_m53 * 0.5;
    const double below_one = 1.0 - two_m53;
    return u < below_one ? u : below_one;
}

// numpy's pairwise float64 sum for n <= 128 (the selector never has more than 64 candidates).
static __device__ double pairwise_small(const double* a, int n) {
    if (n < 8) {
        double r = 0.0;
        for (int i = 0; i < n; ++i) r += a[i];
        return r;
    }
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int i = 8;
    for (; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
}

// One keyed choice among `k` (<= 64) scored candidates, as `try_choose` with top_k = k: order by (score desc, id asc),
// temperature, top_p, min_p, then the Gumbel argmax with the draw keyed on the draft's own absolute position.
// Returns the index of the chosen candidate. NaN scores sort last and are never chosen unless every score is NaN.
// `margin` (nullable) receives how far the winner is ahead of the runner-up in the quantity that was maximised (the noised score in
// temperature units, or the raw score gap at temperature 0): the target agrees with this pick only if its own noised scores keep
// that order, so a large margin is a confident draft and a small one a coin flip. 30 when only one candidate is in play.
static __device__ int keyed_choose(const float* scores, const int* c_row, int k, unsigned long long seed, double temperature, double top_p,
                            double min_p, unsigned long long position, float* margin = nullptr) {
    int ord[64];
    for (int i = 0; i < k; ++i) {
        int j = i;
        while (j > 0) {
            const float va = scores[ord[j - 1]], vb = scores[i];
            const bool before = (vb > va) || (vb == va && c_row[i] < c_row[ord[j - 1]]) || (va != va && vb == vb);
            if (!before) break;
            ord[j] = ord[j - 1];
            --j;
        }
        ord[j] = i;
    }
    if (temperature == 0.0) {
        if (margin) *margin = k > 1 ? scores[ord[0]] - scores[ord[1]] : 30.f;
        return ord[0];
    }
    const double t = temperature > 1e-6 ? temperature : 1e-6;
    double sc[64], pr[64];
    int keep = k;
    for (int i = 0; i < k; ++i) sc[i] = (double)scores[ord[i]] / t;
    if (top_p > 0.0 && top_p < 1.0) {
        const double mx = sc[0];
        for (int i = 0; i < k; ++i) pr[i] = exp(sc[i] - mx);
        const double total = pairwise_small(pr, k);
        double c = 0.0;
        int below = 0;
        for (int i = 0; i < k; ++i) {
            c += pr[i] / total;
            if (c < top_p) ++below;
        }
        keep = below + 1 < k ? below + 1 : k;
    }
    if (min_p > 0.0) {
        const double floor_v = sc[0] + log(min_p);
        int n = 0;
        for (int i = 0; i < keep; ++i)
            if (sc[i] >= floor_v) ++n;
        keep = n > 1 ? n : 1;
    }
    int best = 0;
    double best_score = -INFINITY, second_score = -INFINITY;
    for (int i = 0; i < keep; ++i) {
        const double u = keyed_uniform(seed, position, (unsigned long long)c_row[ord[i]]);
        const double score = sc[i] + (-log(-log(u)));
        if (score > best_score) {
            second_score = best_score;
            best_score = score;
            best = i;
        } else if (score > second_score) {
            second_score = score;
        }
    }
    if (margin) *margin = keep > 1 ? (float)fmin(best_score - second_score, 30.0) : 30.f;
    return ord[best];
}

}  // namespace gb_keyed
