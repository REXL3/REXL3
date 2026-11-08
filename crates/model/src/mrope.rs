//! Interleaved M-RoPE positions for a prompt with images (`docs/VISION.md`, HF `get_rope_index`, ExLlamaV3 `gen_mrope_pos_ids`).
//!
//! Text tokens advance all three position streams together. An image's placeholder run is laid out on its merged patch grid
//! (`rows x cols`): token k sits at (T, H, W) = (b, b + k / cols, b + k % cols) where `b` is the next free position, and the next
//! text token continues at `b + max(rows, cols)`, so an image occupies far fewer positions than it has tokens. The KV cache
//! index stays the token index; only RoPE sees these positions.

/// One image's placeholder run: first token index, token count and merged grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub start: usize,
    pub tokens: usize,
    pub rows: usize,
    pub cols: usize,
}

/// (T, H, W) rope positions of every token of a prompt of `n` tokens, and the next free position after it (the rope position
/// of the first generated token). `runs` must be sorted, non-overlapping, inside `n`, with `tokens == rows * cols`.
pub fn positions(n: usize, runs: &[Run]) -> (Vec<[i32; 3]>, i32) {
    let mut out = Vec::with_capacity(n);
    let mut next = 0i32;
    let mut t = 0usize;
    for r in runs {
        assert!(r.start >= t && r.start + r.tokens <= n && r.tokens == r.rows * r.cols, "malformed image run {r:?}");
        while t < r.start {
            out.push([next; 3]);
            next += 1;
            t += 1;
        }
        for k in 0..r.tokens {
            out.push([next, next + (k / r.cols) as i32, next + (k % r.cols) as i32]);
        }
        next += r.rows.max(r.cols) as i32;
        t += r.tokens;
    }
    while t < n {
        out.push([next; 3]);
        next += 1;
        t += 1;
    }
    (out, next)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_is_the_identity_with_no_delta() {
        let (p, next) = positions(5, &[]);
        assert_eq!(p, vec![[0; 3], [1; 3], [2; 3], [3; 3], [4; 3]]);
        assert_eq!(next, 5);
    }

    #[test]
    fn an_image_is_laid_out_on_its_grid_and_later_text_resumes_behind_the_token_index() {
        // 2 text tokens, a 2 x 3 image (6 tokens), 2 text tokens: 10 tokens
        let (p, next) = positions(10, &[Run { start: 2, tokens: 6, rows: 2, cols: 3 }]);
        assert_eq!(&p[..2], &[[0; 3], [1; 3]]);
        // base 2: rows 0..2, cols 0..3
        assert_eq!(&p[2..8], &[[2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4]]);
        // the next text token continues at 2 + max(2, 3) = 5, although its token index is 8
        assert_eq!(&p[8..], &[[5; 3], [6; 3]]);
        assert_eq!(next, 7);
        assert_eq!(next - 10, -3, "rope delta for the generated tokens");
    }

    #[test]
    fn two_images_chain_their_bases() {
        let runs = [Run { start: 1, tokens: 4, rows: 2, cols: 2 }, Run { start: 6, tokens: 1, rows: 1, cols: 1 }];
        let (p, next) = positions(8, &runs);
        assert_eq!(p[0], [0; 3]);
        assert_eq!(&p[1..5], &[[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2]]);
        assert_eq!(p[5], [3; 3]); // 1 + max(2, 2) = 3
        assert_eq!(p[6], [4, 4, 4]); // a 1x1 image: base 4
        assert_eq!(p[7], [5; 3]);
        assert_eq!(next, 6);
    }
}
