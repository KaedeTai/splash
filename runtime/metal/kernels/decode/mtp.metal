#include "metal/abi/KernelABI.h"
#include "metal/abi/Mtp.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

constant constexpr uint kNoIndex = 0xFFFFFFFFu;

// Qwen3.8-Flash-Next's MTP head (metal/abi/Mtp.h, model/Qwen4ExpTarget.cpp):
// the steps around its layer, which runs on the target's kernels. Rows are a
// decode step's, SPLASH_TARGET_VERIFY_ROWS a lane.

// A draft step's tokens: every row of a lane takes the lane's anchor (step
// 0, row 0 of its draft input rows) or its previous step's proposal.
kernel void mtp_step_tokens(device const uint *draft_input [[buffer(0)]],
                            device const uint *proposed [[buffer(1)]], device uint *tokens [[buffer(2)]],
                            constant MtpRowsParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  if (index >= p.lanes * p.lane_rows) return;
  const uint lane = index / p.lane_rows;
  tokens[index] = p.step == 0 ? draft_input[lane * p.lane_rows]
                              : proposed[lane * SPLASH_DRAFT_PROPOSAL_TOKENS + p.step - 1];
}

// The head's norm of a row's streams, one RMS over all of their values
// (mlx-vlm's pre_fc_norm_hidden), times its F32 weights, into the bf16 rows
// of stream s at normalized + s * rows * width: each stream's rows are one
// projection's input.
kernel void mtp_fuse_norm(device const float *streams [[buffer(0)]], device const float *weight [[buffer(1)]],
                          device bfloat *normalized [[buffer(2)]], constant MtpRowsParams &p [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[MTP_THREADS / 32];
  const uint values = p.width * p.streams;
  device const float *x = streams + ulong(row) * values;
  float sum = 0.0f;
  for (uint i = tid; i < values; i += MTP_THREADS) sum += x[i] * x[i];
  const float inverse = rms_inverse_of_sums<MTP_THREADS / 32>(sum, values, reductions, tid, lane, sg);
  for (uint i = tid; i < values; i += MTP_THREADS) {
    const uint stream = i / p.width, column = i % p.width;
    normalized[(ulong(stream) * p.stream_rows + row) * p.width + column] = bfloat(x[i] * inverse * weight[i]);
  }
}

// The head's streams: stream s of a row its projected stream plus the
// row's projected embedding.
kernel void mtp_fuse_combine(device const bfloat *projected [[buffer(0)]],
                             device const bfloat *embedding [[buffer(1)]], device float *streams [[buffer(2)]],
                             constant MtpRowsParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  const uint rows = p.lanes * p.lane_rows;
  if (index >= rows * p.streams * p.width) return;
  const uint row = index / (p.streams * p.width), rest = index % (p.streams * p.width);
  const uint stream = rest / p.width, column = rest % p.width;
  streams[index] = float(projected[(ulong(stream) * p.stream_rows + row) * p.width + column]) +
                   float(embedding[ulong(row) * p.width + column]);
}

// Row `from_row` of every lane's rows of `source` into row 0 of the lane's
// rows of `destination` (or, with shift, rows 1.. from rows 0..): a lane of
// lane_rows rows of `values` floats.
kernel void mtp_copy_rows(device const float *source [[buffer(0)]], device float *destination [[buffer(1)]],
                          constant uint4 &p [[buffer(2)]], uint index [[thread_position_in_grid]]) {
  // p: x rows to copy, y source row, z destination row, w values a row.
  if (index >= p.x * p.w) return;
  const uint row = index / p.w, value = index % p.w;
  destination[ulong(p.z + row) * p.w + value] = source[ulong(p.y + row) * p.w + value];
}

// A lane's last committed streams: row retained - 1 of its verify rows.
kernel void mtp_save_last(device const float *rows [[buffer(0)]], device const uint *retained [[buffer(1)]],
                          device float *last [[buffer(2)]], constant uint &values [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  if (index >= values) return;
  const uint row = max(retained[0], 1u) - 1;
  last[index] = rows[ulong(row) * values + index];
}

// One lane's attention of its rows over its ring and its own rows, in the
// verify chunk staging layout the target's kernels use
// (kernels/common/attention_qkv_prepare.h): queries
// [kv head][stride][group][head dimension], keys [kv head][stride][head
// dimension], values [kv head][head dimension][stride], and the output as
// the queries. A threadgroup per row and query head; its simdgroups take
// every eighth position, each keeping its own softmax, which thread 0 of
// each simdgroup then merges in order.
kernel void mtp_attention(device const bfloat *queries [[buffer(0)]], device const bfloat *chunk_keys [[buffer(1)]],
                          device const bfloat *chunk_values [[buffer(2)]],
                          device const bfloat *ring_keys [[buffer(3)]],
                          device const bfloat *ring_values [[buffer(4)]], device bfloat *output [[buffer(5)]],
                          constant MtpAttentionParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint D = MTP_HEAD_DIMENSION, Group = MTP_QUERY_HEADS / MTP_KV_HEADS, Simdgroups = MTP_THREADS / 32;
  constexpr uint PerLane = D / 32;
  const uint row = group.x, head = group.y;
  if (row >= p.rows) return;
  const uint kv = head / Group, local = head % Group;
  const uint position = p.first + row;
  // The ring's positions the row reads: [low, first).
  const uint window_low = position + 1 > MTP_ATTENDED_POSITIONS ? position + 1 - MTP_ATTENDED_POSITIONS : 0;
  const uint low = min(max(window_low, p.valid_from), p.first);
  const uint history = p.first - low, keys = history + row + 1;

  device const bfloat *q = queries + ((ulong(kv) * p.stride + row) * Group + local) * D;
  float qv[PerLane];
  for (uint i = 0; i < PerLane; ++i) qv[i] = float(q[lane * PerLane + i]) * p.scale;

  float maximum = -INFINITY, sum = 0.0f, acc[PerLane];
  for (uint i = 0; i < PerLane; ++i) acc[i] = 0.0f;
  for (uint t = sg; t < keys; t += Simdgroups) {
    float k[PerLane], v[PerLane];
    if (t < history) {
      const uint slot = (low + t) % MTP_RING_POSITIONS;
      device const bfloat *kr = ring_keys + (ulong(kv) * MTP_RING_POSITIONS + slot) * D + lane * PerLane;
      device const bfloat *vr = ring_values + (ulong(kv) * MTP_RING_POSITIONS + slot) * D + lane * PerLane;
      for (uint i = 0; i < PerLane; ++i) {
        k[i] = float(kr[i]);
        v[i] = float(vr[i]);
      }
    } else {
      const uint own = t - history;
      device const bfloat *kr = chunk_keys + (ulong(kv) * p.stride + own) * D + lane * PerLane;
      for (uint i = 0; i < PerLane; ++i) {
        k[i] = float(kr[i]);
        v[i] = float(chunk_values[(ulong(kv) * D + lane * PerLane + i) * p.stride + own]);
      }
    }
    float dot = 0.0f;
    for (uint i = 0; i < PerLane; ++i) dot += qv[i] * k[i];
    dot = simd_sum(dot);
    const float next = max(maximum, dot);
    const float rescale = exp(maximum - next), weight = exp(dot - next);
    sum = sum * rescale + weight;
    for (uint i = 0; i < PerLane; ++i) acc[i] = acc[i] * rescale + weight * v[i];
    maximum = next;
  }
  threadgroup float maxima[Simdgroups], sums[Simdgroups];
  threadgroup float partial[Simdgroups][D];
  if (lane == 0) {
    maxima[sg] = maximum;
    sums[sg] = sum;
  }
  for (uint i = 0; i < PerLane; ++i) partial[sg][lane * PerLane + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Thread d of the threadgroup merges dimension d.
  float total_maximum = -INFINITY;
  for (uint s = 0; s < Simdgroups; ++s) total_maximum = max(total_maximum, maxima[s]);
  float total = 0.0f, value = 0.0f;
  for (uint s = 0; s < Simdgroups; ++s) {
    if (sums[s] == 0.0f) continue;
    const float weight = exp(maxima[s] - total_maximum);
    total += sums[s] * weight;
    value += partial[s][tid] * weight;
  }
  output[((ulong(kv) * p.stride + row) * Group + local) * D + tid] = bfloat(total > 0.0f ? value / total : 0.0f);
}

// One lane's rows' keys and values from the verify chunk staging into its
// ring, at their positions.
kernel void mtp_ring_store(device const bfloat *chunk_keys [[buffer(0)]],
                           device const bfloat *chunk_values [[buffer(1)]], device bfloat *ring_keys [[buffer(2)]],
                           device bfloat *ring_values [[buffer(3)]], constant MtpAttentionParams &p [[buffer(4)]],
                           uint index [[thread_position_in_grid]]) {
  constexpr uint D = MTP_HEAD_DIMENSION;
  if (index >= p.rows * MTP_KV_HEADS * D) return;
  const uint row = index / (MTP_KV_HEADS * D), rest = index % (MTP_KV_HEADS * D);
  const uint kv = rest / D, d = rest % D;
  const uint slot = (p.first + row) % MTP_RING_POSITIONS;
  ring_keys[(ulong(kv) * MTP_RING_POSITIONS + slot) * D + d] = chunk_keys[(ulong(kv) * p.stride + row) * D + d];
  ring_values[(ulong(kv) * MTP_RING_POSITIONS + slot) * D + d] = chunk_values[(ulong(kv) * D + d) * p.stride + row];
}

// Each lane's row 0 of the logits by MTP_ARGMAX_SHARDS shards: a shard's
// maximum, its first index and its sum of exp(logit - maximum).
kernel void mtp_argmax_partial(device const float *logits [[buffer(0)]],
                               device MtpArgmaxPartial *partials [[buffer(1)]], constant MtpRowsParams &p [[buffer(2)]],
                               uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint shard = group.x, row_lane = group.y;
  const uint span = (p.vocabulary + MTP_ARGMAX_SHARDS - 1) / MTP_ARGMAX_SHARDS;
  const uint begin = shard * span, end = min(begin + span, p.vocabulary);
  device const float *row = logits + ulong(row_lane) * p.lane_rows * p.vocabulary;
  float maximum = -INFINITY, sum = 0.0f;
  uint index = kNoIndex;
  for (uint i = begin + tid; i < end; i += MTP_THREADS) {
    const float value = row[i];
    if (!isfinite(value)) continue;
    if (value > maximum) {
      sum = sum * exp(maximum - value) + 1.0f;
      maximum = value;
      index = i;
    } else {
      sum += exp(value - maximum);
    }
  }
  threadgroup float maxima[MTP_THREADS], sums[MTP_THREADS];
  threadgroup uint indices[MTP_THREADS];
  maxima[tid] = maximum;
  sums[tid] = sum;
  indices[tid] = index;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float m = -INFINITY, s = 0.0f;
  uint best = kNoIndex;
  for (uint t = 0; t < MTP_THREADS; ++t) {
    if (indices[t] == kNoIndex) continue;
    if (maxima[t] > m || (maxima[t] == m && indices[t] < best)) {
      s = s * exp(m - maxima[t]) + sums[t];
      m = maxima[t];
      best = indices[t];
    } else {
      s += sums[t] * exp(maxima[t] - m);
    }
  }
  partials[row_lane * MTP_ARGMAX_SHARDS + shard] = {m, best, s};
}

// Each lane's proposal at the step, the argmax of its shards, and its
// confidence, the softmax probability of that token.
kernel void mtp_argmax_final(device const MtpArgmaxPartial *partials [[buffer(0)]],
                             device uint *proposed [[buffer(1)]], device float *confidence [[buffer(2)]],
                             constant MtpRowsParams &p [[buffer(3)]], uint lane [[thread_position_in_grid]]) {
  if (lane >= p.lanes) return;
  float m = -INFINITY, s = 0.0f;
  uint best = 0;
  bool found = false;
  for (uint shard = 0; shard < MTP_ARGMAX_SHARDS; ++shard) {
    const MtpArgmaxPartial partial = partials[lane * MTP_ARGMAX_SHARDS + shard];
    if (partial.index == kNoIndex) continue;
    if (!found || partial.maximum > m) {
      s = found ? s * exp(m - partial.maximum) + partial.sum : partial.sum;
      m = partial.maximum;
      best = partial.index;
      found = true;
    } else {
      s += partial.sum * exp(partial.maximum - m);
    }
  }
  proposed[lane * SPLASH_DRAFT_PROPOSAL_TOKENS + p.step] = best;
  confidence[lane * SPLASH_DRAFT_PROPOSAL_TOKENS + p.step] = found && s > 0.0f ? 1.0f / s : 0.0f;
}

// Every lane's proposals and confidences past the steps it drafts.
kernel void mtp_proposal_tail(device uint *proposed [[buffer(0)]], device float *confidence [[buffer(1)]],
                              constant MtpRowsParams &p [[buffer(2)]], uint index [[thread_position_in_grid]]) {
  if (index >= p.lanes * SPLASH_DRAFT_PROPOSAL_TOKENS) return;
  if (index % SPLASH_DRAFT_PROPOSAL_TOKENS < p.step) return;
  proposed[index] = 0;
  confidence[index] = 0.0f;
}
