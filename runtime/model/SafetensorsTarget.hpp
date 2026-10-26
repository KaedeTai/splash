#pragma once

// Source adapter for a safetensors target (MLX's, Model Optimizer's or
// compressed-tensors'): its block images (model/SafetensorsImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/ExpertStore.hpp"
#include "model/GgufImage.hpp"
#include "model/GgufTarget.hpp"
#include "model/SparseRowStore.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace splash::model {

class SafetensorsTargetLoader final {
public:
  // Plans every image of the checkpoint in directory once. With expert
  // memory (experts.memoryBytes), a MoE target's routed experts stream
  // (streamedExperts()) instead of loading with their layers. With a table
  // store (table.cacheBytes), the rows of the table a family's checkpoint
  // holds (Flash-Next's PLE n-gram table) stream from its shards through
  // it (streamTable()) instead of loading (table()).
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const QwenTargetDimensions &geometry, ExpertStoreOptions experts = {},
                          SparseRowStoreOptions table = {});
  SafetensorsTargetLoader(const SafetensorsTargetLoader &) = delete;
  SafetensorsTargetLoader &operator=(const SafetensorsTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The image of the table a family's checkpoint holds beside its layers
  // (safetensors::planImages: Qwen3.8-Flash-Next's PLE n-gram table), which
  // loads with the weights; refused for a target without one.
  [[nodiscard]] WeightFile table();
  // The streamed table, its store reading the checkpoint's shards; none when
  // the table loads.
  [[nodiscard]] std::optional<GgufTargetLoader::StreamedTable> streamTable() const;
  // The image of the MTP head the checkpoint embeds (safetensors::hasMtp),
  // which loads with the weights; refused for a target without one.
  [[nodiscard]] bool hasMtp() const noexcept { return planned_->mtp != 0; }
  [[nodiscard]] WeightFile mtp();
  // The checkpoint, for what a family reads beside its images (Flash-Next's
  // PLE hash).
  [[nodiscard]] const SafetensorsCheckpoint &checkpoint() const noexcept { return planned_->checkpoint; }
  // Whether its images hold F32 norms (safetensors::float32Norms).
  [[nodiscard]] bool float32Norms() const noexcept { return float32Norms_; }
  // Whether the routed experts stream, then the store's options and every
  // layer's experts, which read the checkpoint's open files.
  [[nodiscard]] bool streamsExperts() const noexcept { return experts_.memoryBytes != 0; }
  [[nodiscard]] const ExpertStoreOptions &expertStoreOptions() const noexcept { return experts_; }
  [[nodiscard]] StreamedBlocks streamedExperts() const;

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images; // layers, head, embedding, a family's table
    uint32_t layers = 0;
    // The index of the table's and the MTP head's images, 0 for none.
    size_t table = 0, mtp = 0;
    SparseRowStoreOptions tableStore;
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  bool float32Norms_ = false;
  ExpertStoreOptions experts_;
};

} // namespace splash::model
