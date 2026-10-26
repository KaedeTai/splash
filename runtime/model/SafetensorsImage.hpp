#pragma once

// Plans the images of a safetensors target, MLX's, Model Optimizer's or
// compressed-tensors': the sections a GGUF target's images hold
// (model/GgufImage.hpp), in their order, which writeGgufImage writes from the
// checkpoint's tensors and BlockTargetFormat reads with the GDN value heads
// grouped, as all keep them. Every quantized tensor keeps the format its
// tensors hold: MLX affine (metal/abi/QuantFormat.h), mxfp4 or nvfp4 as MLX
// infers it, or NVFP4 or FP8 as the other two store them: projections and
// experts as planes, a quantized token table as native rows. An unquantized
// token table is copied as its bf16 rows. Norms are bf16 as MLX stores them, or the F32 1 + w of the w
// transformers stores; the convolution and dt_bias are copied as stored, and
// the GDN decay is float(-exp(double(A_log))). The MoE router and the
// shared-expert gate, which the block kernels read in F32, become the F32
// values of their quantization (or of their bf16 weights), and so do GDN
// alpha and beta unless both are quantized in one format.

#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <string>
#include <vector>

namespace splash::model::safetensors {

// The names a checkpoint's language-model modules take: MLX's (mlx-lm,
// mlx-vlm: language_model.model.*, the routed experts stacked under
// mlp.switch_mlp), or transformers' (as NVIDIA's Model Optimizer keeps them:
// model.language_model.*, lm_head, each routed expert a module mlp.experts.<e>
// of its own), whose RMSNorm weights are stored 1 below the weights the norm
// multiplies by, but the GDN's gated norm's.
enum class ModuleNames : uint8_t { Mlx, Transformers };

// The names the checkpoint's tensors take: transformers' when it holds
// lm_head.weight (or compressed-tensors' NVFP4 lm_head.weight_packed).
[[nodiscard]] ModuleNames moduleNames(const SafetensorsCheckpoint &checkpoint);

// Whether the images hold the norms as F32, as they hold a checkpoint's of
// transformers names (their 1 + w, the GDN's gated norm widened), rather than
// MLX's bf16 norms as stored. A hyper-connected target's (Qwen3.8-Flash-Next)
// are F32 by either names: MLX stores its RMSNorm weights 1 below the
// weights the norm multiplies by too (mlx-vlm's Qwen4ExpRMSNorm applies
// 1 + w), but the GDN's gated norm's, and its hyper-connection kernels read
// F32 norms (model/Qwen4Exp.cpp readHyper).
[[nodiscard]] constexpr bool float32Norms(ModuleNames names, const QwenTargetDimensions &geometry) noexcept {
  return names == ModuleNames::Transformers || geometry.hyperConnected();
}

// The layers' images, then the head's and the embedding's, then, for a
// target with a PLE n-gram embedding (Qwen3.8-Flash-Next), its table's
// (ple.bin), as a GGUF's (model/GgufImage.hpp): MLX's row shards
// (ngram_embedding.shards.<s>, mlx-vlm's ShardedEmbedding) back to back as
// one table of native rows, each row's values scaled by the table's
// weight_scale, which the writer folds into an affine row's scales and
// biases. The checkpoint outlives them: their rows read its tensors.
[[nodiscard]] std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint,
                                                  const QwenTargetDimensions &geometry,
                                                  gguf::PlanOptions options = {});

// The modules the images read only quantized, in their order and by `names`:
// the layers' projections, the routed experts' (as a configuration names
// them: MLX's stacked tensor per projection, once transformers' mlp.experts) and
// the shared expert's, and the head. planImages refuses a checkpoint holding
// one unquantized, and the configuration check one its config.json states
// unquantized (model/ModelDescriptor.mm). The router, the shared-expert gate,
// GDN alpha and beta and the token table are not among them.
[[nodiscard]] std::vector<std::string> quantizedModules(const QwenTargetDimensions &geometry, ModuleNames names);

// Whether the images end with the image of the MTP head an MLX checkpoint
// of a hyper-connected MoE target embeds (mtp.*, Qwen3.8-Flash-Next's),
// after the table's: unless SPLASH_MTP=0 leaves it out (mtpDisabled).
[[nodiscard]] bool hasMtp(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry);
[[nodiscard]] bool mtpDisabled();

// The MLX names of the PLE n-gram embedding of a target that has one
// (QwenTargetDimensions::hasPle): the module under which its hash's tensors
// (layer_multipliers, ngram_heads_offsets, ngram_heads_vocab_sizes) and its
// table (ngram_embedding) are, and the table's row shards' modules, which
// the checkpoint holds from shards.0 on, and their rows.
[[nodiscard]] std::string pleEmbeddingModule(const QwenTargetDimensions &geometry, ModuleNames names);
struct PleShard final {
  std::string module;
  uint64_t rows = 0;
};
[[nodiscard]] std::vector<PleShard> pleShards(const SafetensorsCheckpoint &checkpoint,
                                              const QwenTargetDimensions &geometry, ModuleNames names);

} // namespace splash::model::safetensors
