// Qwen3.8-Flash-Next's embedded MTP head as a draft (model/Qwen4Exp.hpp
// Qwen4MtpWeights, kernels/decode/mtp.metal). A lane's head drafts its next
// tokens one at a time before a verify: step i reads the streams of the
// position before it (the target's last committed row's for step 0, its
// own step's after) and the token at it (the anchor, then its own
// proposals), runs its one layer at the position anchor - 1 + i and its
// head, and proposes the argmax. The layer attends a ring of its own keys
// and values of the positions before (MTP_RING_POSITIONS a lane), without
// the QSA indexer the checkpoint holds, and the step's own: the draft only
// shapes which tokens the target verifies, never what it emits. After every
// verify's commit, whichever draft the cycle ran, the catch-up computes the
// ring's entries again from the target's streams and the verify's tokens:
// those of the positions from anchor - 1 on, of which a later step reads
// only those the lane retained (the rest it writes again first).

#include "model/Qwen4Exp.hpp"
#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/HyperConnection.h"
#include "metal/abi/Mtp.h"
#include "metal/abi/PagedAttention.h"
#include "ops/Embedding.hpp"
#include "ops/HyperConnection.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RoPE.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kRows = ExecutionLimits::targetVerifyRows;
constexpr uint32_t kLanes = ExecutionLimits::maximumBatchWidth;
// The RoPE rows of a cycle's lanes, and the rows a draft step past its first
// reads after a lane's last (its view starts that many rows on).
constexpr uint32_t kRopeRows = kLanes * kRows + kRows;
constexpr uint64_t kRingTensorBytes = uint64_t{MTP_KV_HEADS} * MTP_RING_POSITIONS * MTP_HEAD_DIMENSION * 2;

static_assert(MTP_HEAD_DIMENSION == MTP_THREADS, "a thread of the attention's merge takes a head dimension");
static_assert(kRows <= SPLASH_VERIFY_CHUNK_STRIDE, "a lane's rows fit its verify chunk staging");

metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes, const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

} // namespace

struct Qwen4ExpTarget::MtpState final {
  struct Lane final {
    metal::MetalBuffer keys;
    metal::MetalBuffer values;
    metal::MetalBuffer last;
  };

  MtpState(metal::MetalBackend &backend, const QwenTargetGeometry &geometry, uint32_t storageLanes)
      : backend(backend), storageRows(storageLanes * kRows), hidden(geometry.hiddenSize),
        streamValues(geometry.streamWidth()) {
    const uint64_t rowBytes = uint64_t{hidden} * 2;
    tokens = zeroed(backend, uint64_t{storageRows} * sizeof(uint32_t), "mtp/tokens");
    embedding = zeroed(backend, storageRows * rowBytes, "mtp/embedding");
    embeddingNormalized = zeroed(backend, storageRows * rowBytes, "mtp/embedding-normalized");
    embeddingProjected = zeroed(backend, storageRows * rowBytes, "mtp/embedding-projected");
    hiddenNormalized = zeroed(backend, geometry.hyperConnections * storageRows * rowBytes, "mtp/hidden-normalized");
    hiddenProjected = zeroed(backend, geometry.hyperConnections * storageRows * rowBytes, "mtp/hidden-projected");
    for (metal::MetalBuffer &buffer : streams)
      buffer = zeroed(backend, uint64_t{storageRows} * streamValues * sizeof(float), "mtp/streams");
    const uint64_t staging = uint64_t{kLanes} * MTP_KV_HEADS * SPLASH_VERIFY_CHUNK_STRIDE * MTP_HEAD_DIMENSION * 2;
    chunkKeys = zeroed(backend, staging, "mtp/chunk-keys");
    chunkValues = zeroed(backend, staging, "mtp/chunk-values");
    ropePositions = zeroed(backend, uint64_t{kRopeRows} * 3 * sizeof(uint32_t), "mtp/rope-positions");
    draftPositions = zeroed(backend, sizeof(uint32_t), "mtp/draft-positions");
    ropeCos = zeroed(backend, uint64_t{kRopeRows} * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "mtp/rope-cos");
    ropeSin = zeroed(backend, uint64_t{kRopeRows} * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "mtp/rope-sin");
    draftRows = zeroed(backend, kLanes * sizeof(uint32_t), "mtp/draft-rows");
    catchUpRows = zeroed(backend, kLanes * sizeof(uint32_t), "mtp/catch-up-rows");
    auto *one = static_cast<uint32_t *>(draftRows.contents());
    auto *all = static_cast<uint32_t *>(catchUpRows.contents());
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      one[lane] = 1;
      all[lane] = kRows;
    }
    argmax = zeroed(backend, uint64_t{kLanes} * MTP_ARGMAX_SHARDS * sizeof(MtpArgmaxPartial), "mtp/argmax");
  }

  // The state lane's ring and last streams, allocated at its first step.
  const Lane &lane(uint32_t stateLane) {
    auto found = lanes.find(stateLane);
    if (found != lanes.end()) return found->second;
    Lane state{zeroed(backend, kRingTensorBytes, "mtp/ring-keys"), zeroed(backend, kRingTensorBytes, "mtp/ring-values"),
               zeroed(backend, uint64_t{streamValues} * sizeof(float), "mtp/last-streams")};
    return lanes.emplace(stateLane, std::move(state)).first->second;
  }

  metal::MetalBackend &backend;
  uint32_t storageRows = 0;
  uint32_t hidden = 0;
  uint32_t streamValues = 0;
  metal::MetalBuffer tokens;
  metal::MetalBuffer embedding;
  metal::MetalBuffer embeddingNormalized;
  metal::MetalBuffer embeddingProjected;
  metal::MetalBuffer hiddenNormalized;
  metal::MetalBuffer hiddenProjected;
  std::array<metal::MetalBuffer, 2> streams;
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  metal::MetalBuffer ropePositions;
  metal::MetalBuffer draftPositions;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer draftRows;
  metal::MetalBuffer catchUpRows;
  metal::MetalBuffer argmax;
  std::map<uint32_t, Lane> lanes;
};

std::shared_ptr<Qwen4ExpTarget::MtpState> Qwen4ExpTarget::makeMtpState(const Qwen4ExpWeights &weights,
                                                                       const QwenTargetGeometry &geometry,
                                                                       metal::MetalBackend &backend,
                                                                       uint32_t storageLanes) {
  if (!weights.mtp) return nullptr;
  if (geometry.attentionQueryHeads != MTP_QUERY_HEADS || geometry.attentionKvHeads != MTP_KV_HEADS ||
      geometry.attentionHeadDimension != MTP_HEAD_DIMENSION || geometry.hiddenSize != HYPER_CONNECTION_NORM_HIDDEN)
    throw std::invalid_argument("the MTP kernels are compiled for Qwen3.8-Flash-Next's attention");
  return std::make_shared<MtpState>(backend, geometry, storageLanes);
}

bool Qwen4ExpTarget::hasMtp() const noexcept { return static_cast<bool>(mtp_); }

const ops::Projection &Qwen4ExpTarget::draftVocabularyProjection() const noexcept {
  return weights_.mtp ? weights_.mtp->head : vocabularyProjection();
}

void Qwen4ExpTarget::prepareMtpLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) {
  if (lane >= kLanes || !position) throw std::invalid_argument("invalid MTP lane");
  auto *positions = static_cast<uint32_t *>(mtp_->ropePositions.contents());
  // Row j of the lane is the position before the anchor's j-th row.
  for (uint32_t row = 0; row < kRows; ++row) {
    const std::array<uint32_t, 3> rotary = request.rotary(position - 1 + row);
    std::copy(rotary.begin(), rotary.end(), positions + (lane * kRows + row) * 3);
  }
}

void Qwen4ExpTarget::addMtpRopeTables(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers,
                                      uint32_t) const {
  MtpState &m = *mtp_;
  ops::RoPE::addTables(graph, m.ropePositions, m.draftPositions, buffers.ropeInverseFrequencies,
                       buffers.ropeInverseFrequencies, m.ropeCos, m.ropeSin, m.ropeCos, m.ropeSin,
                       {.target_rows = kRopeRows, .draft_rows = 0,
                        .target_scale = geometry_.ropeScaling.attentionFactor()},
                       kRopeRows);
}

ops::PreparedInput Qwen4ExpTarget::addMtpMix(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &b,
                                             const Qwen4Scratch &h, metal::MetalBuffer streams,
                                             const Qwen4HyperWeights &hyper, metal::MetalBuffer output,
                                             uint32_t lanes, const ops::Projection *next) const {
  const ops::HyperShape hs{geometry_.hiddenSize, geometry_.hyperConnections, geometry_.hyperRank};
  const uint32_t rows = lanes * kRows;
  const ops::LinearScratch scratch = b.linearScratch;
  ops::HyperConnection::addNorm(graph, streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs, rows);
  addDecodeProjection(graph, h.normalized, hyper.down, h.low, lanes, scratch);
  ops::PreparedInput prepared =
      ops::HyperConnection::addLow(graph, h.low, hs, rows, scratch, decodeInput(hyper.up, lanes));
  addDecodeProjection(graph, h.low, hyper.up, h.gate, lanes, scratch, std::move(prepared));
  return ops::HyperConnection::addMix(graph, h.normalized, h.gate, std::move(output), hs, rows, scratch,
                                      next ? decodeInput(*next, lanes) : ops::LinearInput::Plain);
}

void Qwen4ExpTarget::addMtpRows(metal::CommandGraph &graph, const QwenMtpBuffers &buffers,
                                std::span<const QwenMtpLane> lanes, const MtpRows &r) const {
  MtpState &m = *mtp_;
  const Qwen4MtpWeights &mtp = *weights_.mtp;
  const QwenTargetVerifyBuffers &b = *buffers.verify;
  const uint32_t width = static_cast<uint32_t>(lanes.size()), rows = width * kRows;
  const uint32_t hidden = geometry_.hiddenSize, streams = geometry_.hyperConnections;
  const Qwen4Scratch h = steps_.verifyScratch(decodeStorageLanes(width) * kRows);
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch scratch = b.linearScratch;
  const MtpRowsParams params{width, kRows, hidden, streams, 0, geometry_.vocabularySize, m.storageRows};
  const uint64_t streamBytes = uint64_t{m.storageRows} * hidden * 2;
  const auto stream = [&](const metal::MetalBuffer &buffer, uint32_t index) {
    return backend_.view(buffer, index * streamBytes, streamBytes);
  };

  // The head's input streams: its projection of the token's embedding added
  // to its projection of each stream of the normalized streams.
  addEmbedding(graph, r.tokens, m.embedding, rows);
  const ops::PreparedInput embedded = ops::Normalization::addRms(
      graph, m.embedding, mtp.embeddingNorm, m.embeddingNormalized, hidden, rows, scratch,
      decodeInput(mtp.embeddingProjection, width));
  addDecodeProjection(graph, m.embeddingNormalized, mtp.embeddingProjection, m.embeddingProjected, width, scratch,
                      embedded);
  graph.add("mtp_fuse_norm", {r.streamsIn, mtp.hiddenNorm.buffer, m.hiddenNormalized}, params, {rows, 1, 1},
            {MTP_THREADS, 1, 1});
  for (uint32_t index = 0; index < streams; ++index)
    addDecodeProjection(graph, stream(m.hiddenNormalized, index), mtp.hiddenProjection,
                        stream(m.hiddenProjected, index), width, scratch);
  graph.add("mtp_fuse_combine", {m.hiddenProjected, m.embeddingProjected, r.streams}, params,
            {(uint64_t{rows} * streams * hidden + MTP_THREADS - 1) / MTP_THREADS, 1, 1}, {MTP_THREADS, 1, 1});

  // The layer's attention, over each lane's ring and its own rows.
  const ops::HyperShape hs{hidden, streams, geometry_.hyperRank};
  const QwenAttentionWeights &attention = mtp.attention;
  const ops::PreparedInput input =
      addMtpMix(graph, b, h, r.streams, mtp.mixerHyper, b.normalized, width, &attention.inputProjection);
  addDecodeProjection(graph, b.normalized, attention.inputProjection, b.fullPacked, width, scratch, input);
  const uint64_t ropeOffset = uint64_t{r.ropeRow} * SPLASH_TARGET_ROPE_PAIRS * sizeof(float);
  const uint64_t ropeBytes = uint64_t{rows} * SPLASH_TARGET_ROPE_PAIRS * sizeof(float);
  ops::PagedAttention::addVerifyProjection(graph, b.fullPacked, attention.queryNorm, attention.keyNorm,
                                           backend_.view(m.ropeCos, ropeOffset, ropeBytes),
                                           backend_.view(m.ropeSin, ropeOffset, ropeBytes), b.fullQueries,
                                           m.chunkKeys, m.chunkValues, MTP_QUERY_HEADS, geometry_.kvLayout, width);
  constexpr uint32_t group = MTP_QUERY_HEADS / MTP_KV_HEADS;
  const uint64_t queryLaneBytes = uint64_t{MTP_KV_HEADS} * SPLASH_VERIFY_CHUNK_STRIDE * group * MTP_HEAD_DIMENSION * 2;
  const uint64_t kvLaneBytes = uint64_t{MTP_KV_HEADS} * SPLASH_VERIFY_CHUNK_STRIDE * MTP_HEAD_DIMENSION * 2;
  for (uint32_t lane = 0; lane < width; ++lane) {
    const QwenMtpLane &l = lanes[lane];
    const MtpState::Lane &state = m.lane(l.stateLane);
    const MtpAttentionParams attend{r.rows, l.position - 1 + r.ropeRow, l.validFrom, SPLASH_VERIFY_CHUNK_STRIDE,
                                    1.0F / std::sqrt(float(MTP_HEAD_DIMENSION))};
    const metal::MetalBuffer keys = backend_.view(m.chunkKeys, lane * kvLaneBytes, kvLaneBytes);
    const metal::MetalBuffer values = backend_.view(m.chunkValues, lane * kvLaneBytes, kvLaneBytes);
    graph.add("mtp_attention",
              {backend_.view(b.fullQueries, lane * queryLaneBytes, queryLaneBytes), keys, values, state.keys,
               state.values, backend_.view(b.fullAttention, lane * queryLaneBytes, queryLaneBytes)},
              attend, {r.rows, MTP_QUERY_HEADS, 1}, {MTP_THREADS, 1, 1});
    graph.add("mtp_ring_store", {keys, values, state.keys, state.values}, attend,
              {(uint64_t{r.rows} * MTP_KV_HEADS * MTP_HEAD_DIMENSION + MTP_THREADS - 1) / MTP_THREADS, 1, 1},
              {MTP_THREADS, 1, 1});
  }
  const ops::LinearPlan outputPlan = linear.decodePlan(attention.outputProjection, width);
  const ops::PreparedInput gated =
      ops::PagedAttention::addVerifyGate(graph, b.fullPacked, b.fullAttention, b.attentionHidden, MTP_QUERY_HEADS,
                                         geometry_.kvLayout, width, scratch, outputPlan.input());
  linear.add(graph, {.input = b.attentionHidden, .output = h.branch, .scratch = scratch, .prepared = gated},
             attention.outputProjection, outputPlan);
  ops::HyperConnection::addCombine(graph, r.streams, h.branch, h.weights, hs, rows);

  // The layer's MoE, of the rows the step runs alone.
  addMtpMix(graph, b, h, r.streams, mtp.ffnHyper, b.normalized, width, nullptr);
  const ops::MoePlan moe = operators_.moeDecode(geometry_.moeShape(), width);
  ops::MoE::add(graph, {b.normalized, h.zeros, h.branch, b.moe, r.rows == 1 ? m.draftRows : m.catchUpRows}, mtp.ffn,
                moe);
  ops::HyperConnection::addCombine(graph, r.streams, h.branch, h.weights, hs, rows);

  if (!r.logits) return;
  const ops::PreparedInput last = addMtpMix(graph, b, h, r.streams, mtp.finalHyper, b.finalHidden, width, &mtp.head);
  addDecodeProjection(graph, b.finalHidden, mtp.head, b.logits, width, scratch, last);
}

void Qwen4ExpTarget::addMtpDraft(metal::CommandGraph &graph, const QwenMtpBuffers &buffers,
                                 std::span<const QwenMtpLane> lanes, uint32_t steps) const {
  const uint32_t width = static_cast<uint32_t>(lanes.size());
  if (!mtp_ || !buffers.verify || !width || width > kLanes || !steps || steps > SPLASH_DRAFT_PROPOSAL_TOKENS)
    throw std::invalid_argument("invalid MTP draft");
  MtpState &m = *mtp_;
  const QwenTargetVerifyBuffers &b = *buffers.verify;
  addMtpRopeTables(graph, b, width);
  MtpRowsParams params{width, kRows, geometry_.hiddenSize, geometry_.hyperConnections, steps,
                       geometry_.vocabularySize, m.storageRows};
  graph.add("mtp_proposal_tail", {buffers.proposedTokens, buffers.confidence}, params, {1, 1, 1},
            {kLanes * SPLASH_DRAFT_PROPOSAL_TOKENS, 1, 1});
  // Step 0 reads each lane's last committed streams in its row 0.
  const uint32_t values = m.streamValues;
  for (uint32_t lane = 0; lane < width; ++lane) {
    if (!lanes[lane].hasLast) throw std::logic_error("an MTP draft lane keeps no last streams");
    const std::array<uint32_t, 4> copy{1, 0, lane * kRows, values};
    graph.add("mtp_copy_rows", {m.lane(lanes[lane].stateLane).last, m.streams[0]}, copy,
              {(values + MTP_THREADS - 1) / MTP_THREADS, 1, 1}, {MTP_THREADS, 1, 1});
  }
  for (uint32_t step = 0; step < steps; ++step) {
    params.step = step;
    graph.add("mtp_step_tokens", {buffers.draftInputTokens, buffers.proposedTokens, m.tokens}, params, {1, 1, 1},
              {width * kRows, 1, 1});
    addMtpRows(graph, buffers, lanes,
               {m.tokens, m.streams[step & 1], m.streams[(step & 1) ^ 1], 1, step, true});
    graph.add("mtp_argmax_partial", {b.logits, m.argmax}, params, {MTP_ARGMAX_SHARDS, width, 1},
              {MTP_THREADS, 1, 1});
    graph.add("mtp_argmax_final", {m.argmax, buffers.proposedTokens, buffers.confidence}, params, {1, 1, 1},
              {width, 1, 1});
  }
}

void Qwen4ExpTarget::addMtpCatchUp(metal::CommandGraph &graph, const QwenMtpBuffers &buffers,
                                   std::span<const QwenMtpLane> lanes) const {
  const uint32_t width = static_cast<uint32_t>(lanes.size());
  if (!mtp_ || !buffers.verify || !width || width > kLanes) throw std::invalid_argument("invalid MTP catch-up");
  MtpState &m = *mtp_;
  const QwenTargetVerifyBuffers &b = *buffers.verify;
  addMtpRopeTables(graph, b, width);
  // The verify's final streams, before the target's last mix.
  const Qwen4Scratch h = steps_.verifyScratch(decodeStorageLanes(width) * kRows);
  const uint32_t values = m.streamValues;
  const auto groups = [&](uint64_t threads) {
    return metal::DispatchSize{(threads + MTP_THREADS - 1) / MTP_THREADS, 1, 1};
  };
  for (uint32_t lane = 0; lane < width; ++lane) {
    const MtpState::Lane &state = m.lane(lanes[lane].stateLane);
    // Row 0, the position before the anchor, reads the lane's last streams;
    // row j its verify row j - 1's.
    if (lanes[lane].hasLast) {
      const std::array<uint32_t, 4> last{1, 0, lane * kRows, values};
      graph.add("mtp_copy_rows", {state.last, m.streams[0]}, last, groups(values), {MTP_THREADS, 1, 1});
    }
    const std::array<uint32_t, 4> shifted{kRows - 1, lane * kRows, lane * kRows + 1, values};
    graph.add("mtp_copy_rows", {h.streams, m.streams[0]}, shifted, groups(uint64_t{kRows - 1} * values),
              {MTP_THREADS, 1, 1});
    graph.add("mtp_save_last",
              {rowsOf<float>(h.streams, lane * kRows, kRows, values),
               backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t)), state.last},
              values, groups(values), {MTP_THREADS, 1, 1});
  }
  addMtpRows(graph, buffers, lanes, {buffers.verifyInputTokens, m.streams[0], m.streams[1], kRows, 0, false});
}

} // namespace splash::model
