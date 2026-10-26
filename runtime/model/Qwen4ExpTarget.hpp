#pragma once

// Qwen3.8-Flash-Next's target (model/Qwen4Exp.hpp): its layers over the
// blocks every Qwen target shares (model/QwenTarget.hpp). Every block reads
// the mix of the fp32 residual streams and adds its output back to each
// stream; the PLE layer adds its gated n-gram values before its mixer's mix;
// the full-attention layers attend through the blocks their QSA indexer
// selects; and the head reads the final mix without a norm. Its step scratch
// and the inputs it makes of a step's tokens are Qwen4Steps'.

#include "model/Qwen4Steps.hpp"
#include "model/QwenTarget.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

struct Qwen4ExpWeights;
struct Qwen4ExpLayerWeights;
struct Qwen4HyperWeights;
struct Qwen4IndexerWeights;

class Qwen4ExpTarget final : public QwenTarget {
public:
  // A target whose prefill commands run up to `prefillChunks` chunks.
  Qwen4ExpTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                 const ops::ExecutionPlans &operators, uint32_t prefillChunks);

  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  void prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) override;
  void addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                 SpilledKvReads &spilled, std::span<const kv::ChunkedPrefillParams> chunks,
                 uint32_t lanes) const override;
  // The hidden rows are the final mix, which the head reads as they are.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const override;
  // The GPU hashes the verify rows' PLE rows, and when the table streams a
  // host step of the command stages them, between the hash and the verify
  // that gathers them.
  void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput, metal::MetalBuffer proposals,
                      metal::MetalBuffer verifyInput, uint32_t lanes) const override;
  // Each lane's PLE history and QSA pending keys after its retained rows,
  // from the convolution inputs and keys its verify wrote.
  void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers, uint32_t lanes) const override;

  // The PLE rows of batch lane `lane`'s verify rows of the last step, each
  // row's heads' in order.
  [[nodiscard]] std::vector<uint32_t> verifyTableRows(uint32_t lane) const override;
  // Its embedded MTP head (model/Qwen4Mtp.cpp), when the weights hold one.
  [[nodiscard]] bool hasMtp() const noexcept override;
  [[nodiscard]] const ops::Projection &draftVocabularyProjection() const noexcept override;
  void addMtpDraft(metal::CommandGraph &graph, const QwenMtpBuffers &buffers, std::span<const QwenMtpLane> lanes,
                   uint32_t steps) const override;
  void addMtpCatchUp(metal::CommandGraph &graph, const QwenMtpBuffers &buffers,
                     std::span<const QwenMtpLane> lanes) const override;

private:
  // The shared prefill's layer loop runs the family's (QwenTarget::addPrefillLayers).
  friend class QwenTarget;

  // A prefill chunk's step: the shared one, the chunk's scratch, the QSA
  // blocks each sequence's rows complete and where each attention layer's
  // QSA index sits in the extents the index tables reach.
  struct PrefillStep : QwenTarget::PrefillStep {
    Qwen4Scratch scratch{};
    std::vector<Qwen4Blocks> blocks{};
    std::span<const uint32_t> indexOffsets{};
  };
  // The family's prefill layers. The streams start as the embedding's rows;
  // every MoE adds its rows to zeros into the branch the next mix adds to the
  // streams; the final mix goes into buffers.hidden[1].
  void addPrefillBegin(PrefillStep &step) const;
  [[nodiscard]] bool attends(uint32_t index) const;
  void addPrefillLayer(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const;
  void addPrefillExperts(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  // A verify step: the shared one, its scratch, the QSA blocks its lanes
  // complete and where each attention layer's QSA index sits.
  struct VerifyStep : QwenTarget::VerifyStep {
    Qwen4Scratch scratch{};
    Qwen4LaneBlocks blocks{};
    std::span<const uint32_t> indexOffsets{};
  };

  // A prefill step's mix of the streams through `hyper` into `output` and,
  // when it injects, the injection weights into the step's scratch.
  void addPrefillMix(PrefillStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer output) const;
  // A verify step's mix into `output`; its projections read the tables its
  // norm and low step write, and with `next`, the decode projection that
  // reads `output` first takes the table the mix writes (the returned input).
  ops::PreparedInput addVerifyMix(VerifyStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer output,
                                  const ops::Projection *next = nullptr) const;
  // The QSA of the attention layer the step reaches next: the indexer's
  // projections of the step's rows and the QSA keys and block index of each
  // sequence or lane (writing its rows' and its completed blocks' entries);
  // a sequence or lane whose rows attend more blocks than they keep then
  // selects theirs, and returns the block lists its attention attends
  // through (none for rows that keep every block).
  [[nodiscard]] std::vector<ops::AttentionBlocks> addPrefillQsa(PrefillStep &step,
                                                                const Qwen4IndexerWeights &indexer) const;
  [[nodiscard]] ops::AttentionBlocks addVerifyQsa(VerifyStep &step, const Qwen4IndexerWeights &indexer) const;
  // The PLE layer's `count` rows into scratch.pleRows: from the table by
  // scratch.pleIndices, or, when the table streams, from the rows staged by
  // scratch.pleSlots, with the table's own gather.
  void addPleGather(metal::CommandGraph &graph, const Qwen4Scratch &scratch, uint32_t count) const;

  // The MTP head's steps (model/Qwen4Mtp.cpp): its buffers, each state
  // lane's ring and last streams, and the rows of a step it runs: from the
  // tokens `tokens` (lane_rows a lane) and the streams `streamsIn`, the
  // head's streams into `streams`, the lanes' `rows` rows from their
  // positions (rope row `ropeRow` on) attending their rings; with `logits`,
  // the head's logits of the step.
  struct MtpState;
  struct MtpRows final {
    metal::MetalBuffer tokens;
    metal::MetalBuffer streamsIn;
    metal::MetalBuffer streams;
    uint32_t rows = 0;
    uint32_t ropeRow = 0;
    bool logits = false;
  };
  void addMtpRows(metal::CommandGraph &graph, const QwenMtpBuffers &buffers, std::span<const QwenMtpLane> lanes,
                  const MtpRows &rows) const;
  ops::PreparedInput addMtpMix(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers,
                               const Qwen4Scratch &scratch, metal::MetalBuffer streams, const Qwen4HyperWeights &hyper,
                               metal::MetalBuffer output, uint32_t lanes, const ops::Projection *next) const;
  void prepareMtpLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request);
  void addMtpRopeTables(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers, uint32_t lanes) const;

  const Qwen4ExpWeights &weights_;
  // The step scratch and the inputs the target makes of a step's tokens.
  Qwen4Steps steps_;
  // Shared, so that its type may stay incomplete here.
  std::shared_ptr<MtpState> mtp_;
  [[nodiscard]] static std::shared_ptr<MtpState> makeMtpState(const Qwen4ExpWeights &weights,
                                                              const QwenTargetGeometry &geometry,
                                                              metal::MetalBackend &backend, uint32_t storageLanes);
};

} // namespace splash::model
