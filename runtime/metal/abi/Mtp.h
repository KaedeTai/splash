#pragma once

// Parameters of the MTP head's draft kernels (kernels/decode/mtp.metal,
// model/Qwen4ExpTarget.cpp): Qwen3.8-Flash-Next's embedded MTP head, which
// drafts a lane's next tokens one at a time from the target's streams, and
// keeps the keys and values of its one attention layer in a ring of its own
// per lane (a window of the positions before the rows it runs).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Threads of the MTP kernels' threadgroups.
#define MTP_THREADS 256u
// Positions a lane's ring holds, and the positions before a row its
// attention reads, fewer by a verify block's rows: a step writes the ring
// up to SPLASH_TARGET_VERIFY_ROWS positions past the ones it keeps.
#define MTP_RING_POSITIONS 2048u
#define MTP_ATTENDED_POSITIONS (MTP_RING_POSITIONS - 16u)
// The head's attention: Qwen3.8-Flash-Next's full attention (24 query heads,
// two KV heads of 256 values).
#define MTP_QUERY_HEADS 24u
#define MTP_KV_HEADS 2u
#define MTP_HEAD_DIMENSION 256u
// Shards of a row's vocabulary the argmax reduces first.
#define MTP_ARGMAX_SHARDS 64u

// A step's rows: `lanes` lanes of `lane_rows` rows each (the verify block's
// rows; a draft step reads row 0 of each lane).
struct MtpRowsParams {
  uint32_t lanes;
  uint32_t lane_rows;
  uint32_t width;   // the hidden size
  uint32_t streams; // streams per row
  uint32_t step;    // the draft step (mtp_step_tokens, mtp_argmax_final)
  uint32_t vocabulary;
  uint32_t stream_rows; // rows a stream's part of the fused norm holds
};

// One lane's attention or ring store: `rows` rows from position `first` on,
// attending the ring's positions from `valid_from` on and its own rows up
// to each, `stride` rows a KV head of the verify chunk staging.
struct MtpAttentionParams {
  uint32_t rows;
  uint32_t first;
  uint32_t valid_from;
  uint32_t stride;
  float scale;
};

// A shard's maximum logit, its first index and the sum of exp(logit - max).
struct MtpArgmaxPartial {
  float maximum;
  uint32_t index;
  float sum;
};
