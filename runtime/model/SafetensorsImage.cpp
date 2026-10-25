#include "model/SafetensorsImage.hpp"

#include "metal/abi/Gguf.h"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <string_view>

namespace splash::model::safetensors {
namespace {

using gguf::Conversion;
using gguf::SafetensorsSource;
using gguf::Repack;
using gguf::TensorRows;

// The prefix of a checkpoint's language-model modules and its head's name,
// by the names it takes (ModuleNames).
std::string modelPrefix(ModuleNames names) {
  return names == ModuleNames::Transformers ? "model.language_model." : "language_model.model.";
}
std::string headName(ModuleNames names) {
  return names == ModuleNames::Transformers ? "lm_head" : "language_model.lm_head";
}

// Plans one image from the checkpoint's tensors, which must have the dtypes
// and shapes its target reads.
class Builder {
public:
  Builder(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry, ModuleNames names,
          std::string name, uint32_t layer, uint32_t type, gguf::PlanOptions options = {})
      : checkpoint_(checkpoint), geometry_(geometry), names_(names), float32Norms_(float32Norms(names, geometry)),
        image_(std::move(name), layer, type, options.streamExperts) {}

  // An RMSNorm's weight, which the norm kernels read (ops::NormWeights): as
  // stored (bf16) from MLX names, or the F32 1 + w of the w transformers
  // stores, and MLX a hyper-connected target's (float32Norms).
  void norm(const std::string &name, uint64_t elements) { normWeight(name, elements, Conversion::CenteredNorm); }

  // The GDN's gated norm, whose weight both store as the norm multiplies by:
  // as stored (bf16) from MLX names, or widened to F32 as the other norms of
  // a transformers checkpoint's images are.
  void gatedNorm(const std::string &name, uint64_t elements) {
    normWeight(name, elements, Conversion::WidenToFloat32);
  }

  // The convolution taps of every channel, as stored: q and k channels, then
  // the value channels grouped by key head. MLX stores the depthwise weight
  // as [channels, taps, 1] and transformers as [channels, 1, taps], which
  // hold the same bytes.
  void convolution(const std::string &name) {
    const uint64_t channels = geometry_.convolutionDimension, taps = kGdnConvolutionTaps;
    stored(name + ".weight", {"BF16"},
           names_ == ModuleNames::Mlx ? std::vector<uint64_t>{channels, taps, 1}
                                      : std::vector<uint64_t>{channels, 1, taps});
  }

  // The bf16 GDN time-step bias of every value head, as stored.
  void timeBias(const std::string &name) { stored(name, {"BF16"}, {geometry_.gdnValueHeads}); }

  // The F32 GDN decay -exp(A_log) of every value head.
  void decay(const std::string &name) {
    const SourceTensor &tensor = require(name, {"BF16", "F32"}, {geometry_.gdnValueHeads});
    image_.copy(rows(name, tensor, tensor.dtype == "BF16" ? ggml::kBF16 : ggml::kF32, 1, tensor.bytes),
                Conversion::Decay);
  }

  // A quantized [rows, columns] tensor, its planes.
  void projection(const std::string &module, uint64_t rows, uint64_t columns) {
    const TensorRows source = requireQuantized(module, rows, columns);
    Repack repack = image_.planes(gguf_format_of(source.type), rows, columns, module);
    repack.sources.push_back(source);
    image_.repack(std::move(repack));
  }

  // The routed experts' [rows, columns] tensors of one projection (gate_proj,
  // up_proj or down_proj) under the MLP prefix `mlp`, as one tensor of
  // experts * rows rows (gguf::ImageBuilder::expertPlanes): MLX's one stacked
  // tensor, or the experts' modules under their per-expert names, which
  // must share a format.
  void experts(const std::string &mlp, const std::string &projection, uint64_t rows, uint64_t columns) {
    const uint64_t count = geometry_.experts;
    if (names_ == ModuleNames::Mlx) {
      const std::string module = mlp + "switch_mlp." + projection;
      const std::optional<TensorRows> source = quantized(module, rows, columns, count);
      if (!source) throw WeightStoreError(module + " is unquantized; Splash loads quantized projections");
      Repack repack = image_.expertPlanes(gguf_format_of(source->type), count * rows, columns, module);
      repack.sources.push_back(*source);
      image_.expertRepack(std::move(repack));
      return;
    }
    std::vector<TensorRows> sources;
    for (uint64_t e = 0; e < count; ++e) {
      sources.push_back(requireQuantized(mlp + "experts." + std::to_string(e) + "." + projection, rows, columns));
      if (sources.back().type != sources.front().type)
        throw WeightStoreError(sources.back().name + " is in another format than its layer's first expert");
    }
    Repack repack = image_.expertPlanes(gguf_format_of(sources.front().type), count * rows, columns, mlp + "experts");
    repack.sources = std::move(sources);
    image_.expertRepack(std::move(repack));
  }

  // beta (value heads rows) | alpha (value heads rows): one tensor of their
  // shared format padded with zero rows to whole plane tiles, or else one F32
  // tensor.
  void alphaBeta(const std::string &beta, const std::string &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    const std::optional<TensorRows> b = quantized(beta, heads, hidden), a = quantized(alpha, heads, hidden);
    if (b && a && b->type == a->type) {
      Repack repack = image_.planes(gguf_format_of(b->type), gguf::tiledRows(2ull * heads), hidden, alpha);
      repack.sources = {*b, *a};
      image_.repack(std::move(repack));
      return;
    }
    const uint64_t rowBytes = uint64_t{hidden} * sizeof(float);
    image_.descriptor(ggml::kF32, 2ull * heads, hidden, {}, {2 * heads * rowBytes, 0, 0}, beta);
    const uint64_t destination = image_.section(2 * heads * rowBytes);
    floatRows(beta, heads, hidden, destination);
    floatRows(alpha, heads, hidden, destination + heads * rowBytes);
  }

  // An F32 [rows, columns] tensor (the MoE router and the shared-expert
  // gate, which the block kernels read unquantized).
  void floatTensor(const std::string &module, uint64_t rows, uint64_t columns) {
    const uint64_t bytes = rows * columns * sizeof(float);
    image_.descriptor(ggml::kF32, rows, columns, {}, {bytes, 0, 0}, module);
    floatRows(module, rows, columns, image_.section(bytes));
  }

  // Rows [first, first + rows) of a [total, columns] module as an F32
  // tensor: the QSA indexer's query and key projections, which MLX keeps as
  // one module (index_qk_proj, the query's rows then the key's) and the
  // indexer reads as two float segments (model/Qwen4Exp.cpp readIndexer).
  void floatSlice(const std::string &module, uint64_t first, uint64_t rows, uint64_t total, uint64_t columns) {
    if (first + rows > total) throw WeightStoreError("rows past the end of " + module);
    const uint64_t bytes = rows * columns * sizeof(float);
    image_.descriptor(ggml::kF32, rows, columns, {}, {bytes, 0, 0}, module);
    const uint64_t destination = image_.section(bytes);
    if (std::optional<TensorRows> source = quantized(module, total, columns)) {
      source->first = first;
      source->rows = rows;
      image_.copyAt(destination, *source, Conversion::DequantizeToFloat32);
      return;
    }
    const SourceTensor &weight = require(module + ".weight", {"BF16", "F32"}, {total, columns});
    const bool bfloat16 = weight.dtype == "BF16";
    TensorRows source =
        this->rows(module, weight, bfloat16 ? ggml::kBF16 : ggml::kF32, total, weight.bytes / total);
    source.first = first;
    source.rows = rows;
    image_.copyAt(destination, source, bfloat16 ? Conversion::WidenToFloat32 : Conversion::None);
  }

  // The F32 rows of a [rows, columns] module alone, without a descriptor
  // (a hyper-connection's injection weights, which the mix reads as F32,
  // model/Qwen4Exp.cpp readHyper).
  void floatValues(const std::string &module, uint64_t rows, uint64_t columns) {
    floatRows(module, rows, columns, image_.section(rows * columns * sizeof(float)));
  }

  // A bf16 tensor of `shape` widened to its F32 values (the PLE's depthwise
  // convolution, [channels, taps, 1] as MLX stores it, channel by channel as
  // a GGUF's ple_conv1d).
  void widened(const std::string &name, const std::vector<uint64_t> &shape) {
    const SourceTensor &tensor = require(name, {"BF16"}, shape);
    image_.copy(rows(name, tensor, ggml::kBF16, 1, tensor.bytes), Conversion::WidenToFloat32);
  }

  // The PLE n-gram table: its shards' native rows back to back after one
  // descriptor of their format, each row scaled by the table's weight_scale
  // (one bf16 or F32 value), which the writer folds into an affine row's
  // scales and biases, as model/Qwen4Exp.cpp reads a GGUF's
  // per_layer_token_embd.
  void pleTable(const std::string &table, const std::vector<PleShard> &shards, uint64_t columns) {
    if (shards.empty()) throw WeightStoreError(table + " holds no row shards");
    std::vector<TensorRows> sources;
    uint64_t rows = 0;
    const SourceTensor *scale = checkpoint_.find(table + ".weight_scale");
    if (scale) scale = &scalar(table + ".weight_scale", {"BF16", "F32"});
    for (const PleShard &shard : shards) {
      std::optional<TensorRows> source = quantized(shard.module, shard.rows, columns);
      if (!source) throw WeightStoreError(shard.module + " is unquantized; Splash loads a quantized PLE table");
      if (!sources.empty() && source->type != sources.front().type)
        throw WeightStoreError(shard.module + " is in another format than the PLE table's first shard");
      if (!quant_affine_format(gguf_format_of(source->type)) && scale)
        throw WeightStoreError(table + " scales rows of a format other than MLX affine");
      if (scale)
        source->scale = {scale->file, scale->offset, scale->bytes, shard.rows, scale->dtype == "BF16", false};
      rows += shard.rows;
      sources.push_back(std::move(*source));
    }
    const uint64_t rowBytes = sources.front().rowBytes;
    image_.descriptor(sources.front().type, rows, columns, {}, {rows * rowBytes, 0, 0}, table);
    uint64_t destination = image_.section(rows * rowBytes);
    for (TensorRows &source : sources) {
      const uint64_t bytes = source.rows * rowBytes;
      image_.copyAt(destination, std::move(source), Conversion::None);
      destination += bytes;
    }
  }

  // The token table's rows, which the embedding kernel gathers: native rows
  // of its quantized format, or its bf16 rows as stored.
  void embedding(const std::string &module) {
    const uint64_t vocabulary = geometry_.vocabularySize, hidden = geometry_.hiddenSize;
    if (const std::optional<TensorRows> source = quantized(module, vocabulary, hidden)) {
      if (!gguf_embedding_format(gguf_format_of(source->type)))
        throw WeightStoreError(module + " is in a format the token gather does not read");
      image_.descriptor(source->type, vocabulary, hidden, {}, {vocabulary * source->rowBytes, 0, 0}, module);
      image_.copy(*source);
      return;
    }
    const SourceTensor &table = require(module + ".weight", {"BF16"}, {vocabulary, hidden});
    image_.descriptor(ggml::kBF16, vocabulary, hidden, {}, {table.bytes, 0, 0}, module);
    image_.copy(rows(module + ".weight", table, ggml::kBF16, vocabulary, table.bytes / vocabulary));
  }

  [[nodiscard]] gguf::Image finish() { return image_.finish(); }

private:
  const SourceTensor &require(const std::string &name, std::initializer_list<std::string_view> dtypes,
                              const std::vector<uint64_t> &shape) const {
    const SourceTensor &tensor = checkpoint_.require(name);
    if (std::find(dtypes.begin(), dtypes.end(), tensor.dtype) == dtypes.end() || tensor.shape != shape)
      throw WeightStoreError("source tensor type or shape does not match: " + name);
    return tensor;
  }

  static TensorRows rows(const std::string &name, const SourceTensor &tensor, uint32_t type, uint64_t count,
                         uint64_t rowBytes) {
    return {name, type, tensor.offset, count, rowBytes, {}, tensor.file};
  }

  // A tensor copied as stored into a section of its own.
  void stored(const std::string &name, std::initializer_list<std::string_view> dtypes, std::vector<uint64_t> shape) {
    const SourceTensor &tensor = require(name, dtypes, shape);
    image_.copy(rows(name, tensor, ggml::kBF16, 1, tensor.bytes));
  }

  // A norm's bf16 weight, as stored from MLX names, or converted to F32 as
  // `transformers` says from transformers names.
  void normWeight(const std::string &name, uint64_t elements, Conversion transformers) {
    if (!float32Norms_) return stored(name + ".weight", {"BF16"}, {elements});
    const SourceTensor &tensor = require(name + ".weight", {"BF16"}, {elements});
    image_.copy(rows(name + ".weight", tensor, ggml::kBF16, 1, tensor.bytes), transformers);
  }

  // The module's native rows, which must be quantized.
  TensorRows requireQuantized(const std::string &module, uint64_t rows, uint64_t columns) const {
    const std::optional<TensorRows> source = quantized(module, rows, columns);
    if (!source) throw WeightStoreError(module + " is unquantized; Splash loads quantized projections");
    return *source;
  }

  // The module's native rows when the checkpoint holds it quantized, in the
  // format its tensors hold (metal/abi/QuantFormat.h); none when the
  // checkpoint holds the module's weight unquantized; a module without a
  // weight is a missing source tensor. MLX's format is the one
  // MLX infers from codes [experts, rows, columns * bits / 32] (U32) and
  // scales [experts, rows, columns / group size]: bf16 with bf16 biases
  // (affine), uint8 E8M0 exponents (mxfp4, 4 bits in groups of 32) or uint8
  // E4M3 scales (nvfp4, 4 bits in groups of 16). Model Optimizer's and
  // compressed-tensors' (floatQuantized) is the one their tensor types state.
  std::optional<TensorRows> quantized(const std::string &module, uint64_t rows, uint64_t columns,
                                      uint64_t experts = 1) const {
    if (experts == 1 && checkpoint_.find(module + ".weight_scale")) return floatQuantized(module, rows, columns);
    const SourceTensor &codes = checkpoint_.require(module + ".weight");
    const SourceTensor *scales = checkpoint_.find(module + ".scales");
    if (!scales) return std::nullopt;
    std::vector<uint64_t> leading{rows};
    if (experts > 1) leading.insert(leading.begin(), experts);
    const auto shaped = [&](const SourceTensor &tensor) {
      return tensor.shape.size() == leading.size() + 1 &&
             std::equal(leading.begin(), leading.end(), tensor.shape.begin()) && tensor.shape.back();
    };
    if (codes.dtype != "U32" || !shaped(codes) || !shaped(*scales) || columns % scales->shape.back() ||
        codes.shape.back() * 32 % columns)
      throw WeightStoreError("source tensor type or shape does not match: " + module);
    const uint64_t group = columns / scales->shape.back(), bits = codes.shape.back() * 32 / columns;
    uint32_t format = GGUF_FMT_COUNT;
    const SourceTensor *biases = nullptr;
    uint32_t groupSplit = 1;
    if (scales->dtype == "BF16") {
      format = quant_affine_format_of(uint32_t(bits), uint32_t(group));
      biases = &require(module + ".biases", {"BF16"}, scales->shape);
      // Groups of 128 whose rows are not whole 256-column plane units are
      // held as twice as many groups of 64 (SafetensorsSource::groupSplit).
      if (format != GGUF_FMT_COUNT && group == 128 && columns % ggufColumnUnit(kQuantFormats[format])) {
        format = quant_affine_format_of(uint32_t(bits), 64);
        groupSplit = 2;
      }
    } else if (scales->dtype == "U8" && bits == 4 && group == 32) {
      format = GGUF_FMT_MXFP4;
    } else if (scales->dtype == "U8" && bits == 4 && group == 16) {
      format = GGUF_FMT_NVFP4;
    }
    if (format == GGUF_FMT_COUNT)
      throw WeightStoreError(module + " is " + std::to_string(bits) + "-bit in groups of " + std::to_string(group) +
                             " with " + scales->dtype + " scales; MLX weights load as affine 2, 3, 4, 5, 6 or 8 "
                             "bits in groups of 32, 64 or 128 with bf16 scales, or as mxfp4 or nvfp4");
    requireWholeBlocks(module, format, columns);
    return TensorRows{module, kQuantFormats[format].ggml_type, 0, experts * rows,
                      ggufRowBytes(kQuantFormats[format], columns), {}, codes.file,
                      {&codes, scales, biases, nullptr, groupSplit}};
  }

  // A module's native rows in NVFP4 or FP8 as Model Optimizer and
  // compressed-tensors store them. NVFP4: U8 codes [rows, columns / 2]
  // (element e in bits 4 (e % 2) of byte e / 2) and F8_E4M3 scales [rows,
  // columns / 16] beside the tensor scale g, Model Optimizer's .weight with g
  // its F32 weight_scale_2, compressed-tensors' .weight_packed with g the
  // reciprocal of its F32 weight_global_scale. FP8: F8_E4M3 values [rows,
  // columns] with an F32 or BF16 weight_scale g, the tensor's or (the
  // channel strategy of compressed-tensors) each row's. Input scales, which
  // scale activations, play no part in weights-only kernels.
  TensorRows floatQuantized(const std::string &module, uint64_t rows, uint64_t columns) const {
    const bool packed = checkpoint_.find(module + ".weight_packed") != nullptr;
    const std::string codes = module + (packed ? ".weight_packed" : ".weight");
    const SourceTensor &weight = checkpoint_.require(codes);
    SafetensorsSource source{&weight};
    const SourceTensor *scale = nullptr;
    bool perRow = false;
    uint32_t format;
    if (weight.dtype == "U8") {
      format = GGUF_FMT_NVFP4;
      require(codes, {"U8"}, {rows, columns / 2});
      source.scales = &require(module + ".weight_scale", {"F8_E4M3"}, {rows, columns / 16});
      scale = &scalar(module + (packed ? ".weight_global_scale" : ".weight_scale_2"), {"F32"});
    } else if (weight.dtype == "F8_E4M3" && !packed) {
      format = GGUF_FMT_FP8;
      require(codes, {"F8_E4M3"}, {rows, columns});
      perRow = checkpoint_.require(module + ".weight_scale").shape == std::vector<uint64_t>{rows, 1};
      scale = perRow ? &require(module + ".weight_scale", {"F32", "BF16"}, {rows, 1})
                     : &scalar(module + ".weight_scale", {"F32", "BF16"});
    } else {
      throw WeightStoreError(codes + " is " + weight.dtype + "; Splash loads NVFP4 (U8 codes) and FP8 (F8_E4M3 "
                             "values) as Model Optimizer and compressed-tensors store them");
    }
    requireWholeBlocks(module, format, columns);
    TensorRows result{module, kQuantFormats[format].ggml_type, 0, rows, ggufRowBytes(kQuantFormats[format], columns),
                      {}, weight.file, source};
    result.scale = {scale->file, scale->offset, scale->bytes, perRow ? 1 : rows, scale->dtype == "BF16", packed};
    return result;
  }

  // A tensor of one value, of shape [] or [1].
  const SourceTensor &scalar(const std::string &name, std::initializer_list<std::string_view> dtypes) const {
    return require(name, dtypes,
                   checkpoint_.require(name).shape.empty() ? std::vector<uint64_t>{} : std::vector<uint64_t>{1});
  }

  // Native rows hold whole native blocks of the module's format; only nvfp4's
  // and fp8's, of 256 elements, are wider than its groups.
  static void requireWholeBlocks(const std::string &module, uint32_t format, uint64_t columns) {
    if (columns % kQuantFormats[format].block_elements)
      throw WeightStoreError(module + "'s rows are not whole " + kQuantFormats[format].name + " blocks of " +
                             std::to_string(kQuantFormats[format].block_elements) + " elements");
  }

  // The F32 values of a [rows, columns] module at `destination`: its
  // quantization's values, or its bf16 or F32 weight.
  void floatRows(const std::string &module, uint64_t rows, uint64_t columns, uint64_t destination) {
    if (const std::optional<TensorRows> source = quantized(module, rows, columns)) {
      image_.copyAt(destination, *source, Conversion::DequantizeToFloat32);
      return;
    }
    const SourceTensor &weight = require(module + ".weight", {"BF16", "F32"}, {rows, columns});
    const bool bfloat16 = weight.dtype == "BF16";
    image_.copyAt(destination, this->rows(module, weight, bfloat16 ? ggml::kBF16 : ggml::kF32, rows, weight.bytes / rows),
                  bfloat16 ? Conversion::WidenToFloat32 : Conversion::None);
  }

  const SafetensorsCheckpoint &checkpoint_;
  const QwenTargetDimensions &geometry_;
  const ModuleNames names_;
  const bool float32Norms_;
  gguf::ImageBuilder image_;
};

// Plans nothing, and names the modules a walk of the images reads only
// quantized (quantizedModules): the projections, a layer's routed experts
// (MLX's stacked tensor of each projection, or once transformers' module
// mlp.experts, which holds them), and the head, which Builder refuses
// unquantized.
struct QuantizedModules {
  explicit QuantizedModules(ModuleNames names) : names(names) {}
  void norm(const std::string &, uint64_t) {}
  void gatedNorm(const std::string &, uint64_t) {}
  void convolution(const std::string &) {}
  void timeBias(const std::string &) {}
  void decay(const std::string &) {}
  void projection(const std::string &module, uint64_t, uint64_t) { modules.push_back(module); }
  void experts(const std::string &mlp, const std::string &projection, uint64_t, uint64_t) {
    const std::string module = names == ModuleNames::Mlx ? mlp + "switch_mlp." + projection : mlp + "experts";
    if (modules.empty() || modules.back() != module) modules.push_back(module);
  }
  void alphaBeta(const std::string &, const std::string &) {}
  void floatTensor(const std::string &, uint64_t, uint64_t) {}
  void floatSlice(const std::string &, uint64_t, uint64_t, uint64_t, uint64_t) {}
  void floatValues(const std::string &, uint64_t, uint64_t) {}
  void widened(const std::string &, const std::vector<uint64_t> &) {}
  void embedding(const std::string &) {}

  ModuleNames names;
  std::vector<std::string> modules;
};

// The sections of each image, in their order, which a Builder plans and
// QuantizedModules names: a layer's, the head's and the embedding's.
//
// A layer's mixer, GDN or full attention, which every family's layers hold.
template <class Walk>
void walkMixer(Walk &b, const QwenTargetDimensions &g, const std::string &prefix, uint32_t index) {
  if (g.isFullAttentionLayer(index)) {
    const std::string attention = prefix + "self_attn.";
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.projection(attention + "q_proj", 2ull * g.attentionWidth, g.hiddenSize);
    b.projection(attention + "k_proj", kvRows, g.hiddenSize);
    b.projection(attention + "v_proj", kvRows, g.hiddenSize);
    b.norm(attention + "q_norm", g.attentionHeadDimension);
    b.norm(attention + "k_norm", g.attentionHeadDimension);
    b.projection(attention + "o_proj", g.hiddenSize, g.attentionWidth);
  } else {
    const std::string gdn = prefix + "linear_attn.";
    b.projection(gdn + "in_proj_qkv", g.convolutionDimension, g.hiddenSize);
    b.projection(gdn + "in_proj_z", g.attentionWidth, g.hiddenSize);
    b.alphaBeta(gdn + "in_proj_b", gdn + "in_proj_a");
    b.convolution(gdn + "conv1d");
    b.decay(gdn + "A_log");
    b.timeBias(gdn + "dt_bias");
    b.gatedNorm(gdn + "norm", g.gdnHeadDimension);
    b.projection(gdn + "out_proj", g.hiddenSize, g.attentionWidth);
  }
}

// A layer's dense FFN or sparse MoE, which every family's layers hold.
template <class Walk> void walkFfn(Walk &b, const QwenTargetDimensions &g, const std::string &prefix) {
  const std::string mlp = prefix + "mlp.";
  if (g.ffnKind == QwenFfnKind::SparseMoe) {
    const uint64_t width = g.expertIntermediateSize;
    b.floatTensor(mlp + "gate", g.experts, g.hiddenSize);
    b.experts(mlp, "gate_proj", width, g.hiddenSize);
    b.experts(mlp, "up_proj", width, g.hiddenSize);
    b.experts(mlp, "down_proj", g.hiddenSize, width);
    b.projection(mlp + "shared_expert.gate_proj", width, g.hiddenSize);
    b.projection(mlp + "shared_expert.up_proj", width, g.hiddenSize);
    b.projection(mlp + "shared_expert.down_proj", g.hiddenSize, width);
    b.floatTensor(mlp + "shared_expert_gate", 1, g.hiddenSize);
  } else {
    b.projection(mlp + "gate_proj", g.intermediateSize, g.hiddenSize);
    b.projection(mlp + "up_proj", g.intermediateSize, g.hiddenSize);
    b.projection(mlp + "down_proj", g.hiddenSize, g.intermediateSize);
  }
}

// One hyper-connection mix, as a GGUF's image holds it (GgufImage.cpp
// hyperImage, model/Qwen4Exp.cpp readHyper): its F32 norm of the streams, its
// low-rank down and up projections, and, for a mix that injects its block's
// output back, its F32 injection weights [hyperConnections, streams].
template <class Walk>
void walkHyper(Walk &b, const QwenTargetDimensions &g, const std::string &mix, bool inject) {
  const uint64_t streams = g.streamWidth();
  b.norm(mix + "hc_norm", streams);
  b.projection(mix + "input_mix_weight_down", g.hyperRank, streams);
  b.projection(mix + "input_mix_weight_up", streams, g.hyperRank);
  if (inject) b.floatValues(mix + "block_inject_weight", g.hyperConnections, streams);
}

// A hyper-connected layer (Qwen3.8-Flash-Next's), as a GGUF's image holds it
// (GgufImage.cpp Qwen4ExpGguf::layer): the mixer's mix, the mixer, the QSA
// indexer of a full-attention layer, the FFN's mix and the FFN, and in the
// PLE's layer its projections, norms and convolution.
template <class Walk>
void walkHyperLayer(Walk &b, const QwenTargetDimensions &g, const std::string &prefix, uint32_t index) {
  walkHyper(b, g, prefix + "attn_hyper_connection.", true);
  walkMixer(b, g, prefix, index);
  if (g.isFullAttentionLayer(index)) {
    const std::string indexer = prefix + "self_attn.indexer.";
    const uint64_t queryRows = uint64_t{g.indexerHeads} * g.indexerHeadDimension, keyRows = g.indexerHeadDimension;
    b.floatSlice(indexer + "index_qk_proj", 0, queryRows, queryRows + keyRows, g.hiddenSize);
    b.floatSlice(indexer + "index_qk_proj", queryRows, keyRows, queryRows + keyRows, g.hiddenSize);
    b.norm(indexer + "q_layernorm", g.indexerHeadDimension);
    b.norm(indexer + "k_layernorm", g.indexerHeadDimension);
  }
  walkHyper(b, g, prefix + "mlp_hyper_connection.", true);
  walkFfn(b, g, prefix);
  if (index != g.pleLayer) return;
  const std::string ple = prefix + "ple.";
  const uint64_t streams = g.streamWidth();
  b.projection(ple + "key_proj", streams, g.pleWidth());
  b.projection(ple + "value_proj", g.hiddenSize, g.pleWidth());
  b.norm(ple + "norm_key", streams);
  b.norm(ple + "norm_query", streams);
  b.norm(ple + "norm_conv", streams);
  b.widened(ple + "conv1d.weight", {streams, g.pleConvolutionTaps, 1});
}

template <class Walk>
void walkLayer(Walk &b, const QwenTargetDimensions &g, ModuleNames names, uint32_t index) {
  const std::string prefix = modelPrefix(names) + "layers." + std::to_string(index) + ".";
  if (g.hyperConnected()) return walkHyperLayer(b, g, prefix, index);
  b.norm(prefix + "input_layernorm", g.hiddenSize);
  walkMixer(b, g, prefix, index);
  b.norm(prefix + "post_attention_layernorm", g.hiddenSize);
  walkFfn(b, g, prefix);
}

// The head: the output norm, or a hyper-connected target's last mix, which
// injects nothing; then the logits.
template <class Walk> void walkHead(Walk &b, const QwenTargetDimensions &g, ModuleNames names) {
  if (g.hyperConnected())
    walkHyper(b, g, modelPrefix(names) + "hyper_connection_mixer.", false);
  else
    b.norm(modelPrefix(names) + "norm", g.hiddenSize);
  b.projection(headName(names), g.vocabularySize, g.hiddenSize);
}

template <class Walk> void walkEmbedding(Walk &b, ModuleNames names) {
  b.embedding(modelPrefix(names) + "embed_tokens");
}

// The header type of a family's table's image (ple.bin), as a GGUF's
// (GgufImage.cpp kTableImageType), beside the layers' 0 and 1 and the head's 2.
constexpr uint32_t kTableImageType = 3;

} // namespace

ModuleNames moduleNames(const SafetensorsCheckpoint &checkpoint) {
  return checkpoint.find("lm_head.weight") || checkpoint.find("lm_head.weight_packed") ? ModuleNames::Transformers
                                                                                         : ModuleNames::Mlx;
}

std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry,
                                   gguf::PlanOptions options) {
  const ModuleNames names = moduleNames(checkpoint);
  std::vector<gguf::Image> images;
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
    Builder b(checkpoint, geometry, names, "layer-" + std::to_string(layer) + ".bin", layer,
              geometry.isFullAttentionLayer(layer) ? 1u : 0u, options);
    walkLayer(b, geometry, names, layer);
    images.push_back(b.finish());
  }
  Builder head(checkpoint, geometry, names, "head.bin", geometry.layers, 2);
  walkHead(head, geometry, names);
  images.push_back(head.finish());
  Builder embedding(checkpoint, geometry, names, "embedding.bin", geometry.vocabularySize, geometry.hiddenSize);
  walkEmbedding(embedding, names);
  images.push_back(embedding.finish());
  if (geometry.hasPle()) {
    Builder table(checkpoint, geometry, names, "ple.bin", geometry.pleLayer, kTableImageType);
    table.pleTable(pleEmbeddingModule(geometry, names) + "ngram_embedding", pleShards(checkpoint, geometry, names),
                   geometry.pleHeadDimension);
    images.push_back(table.finish());
  }
  return images;
}

std::vector<std::string> quantizedModules(const QwenTargetDimensions &geometry, ModuleNames names) {
  QuantizedModules walk(names);
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) walkLayer(walk, geometry, names, layer);
  walkHead(walk, geometry, names);
  walkEmbedding(walk, names);
  return std::move(walk.modules);
}

std::string pleEmbeddingModule(const QwenTargetDimensions &geometry, ModuleNames names) {
  return modelPrefix(names) + "layers." + std::to_string(geometry.pleLayer) + ".ple.ple_embedding.";
}

std::vector<PleShard> pleShards(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry,
                                ModuleNames names) {
  // mlx-vlm names shard s ngram_embedding.shards.<s>; a conversion that kept
  // transformers' spelling names it ngram_embedding.shard_<s>.
  const std::string table = pleEmbeddingModule(geometry, names) + "ngram_embedding.";
  std::vector<PleShard> shards;
  for (uint32_t shard = 0;; ++shard) {
    std::string module = table + "shards." + std::to_string(shard);
    if (!checkpoint.find(module + ".weight")) module = table + "shard_" + std::to_string(shard);
    const SourceTensor *weight = checkpoint.find(module + ".weight");
    if (!weight) break;
    if (weight->shape.size() != 2 || !weight->shape[0])
      throw WeightStoreError("source tensor type or shape does not match: " + module + ".weight");
    shards.push_back({std::move(module), weight->shape[0]});
  }
  if (shards.empty()) throw WeightStoreError("the checkpoint holds no PLE table: " + table + "shards.0");
  return shards;
}

} // namespace splash::model::safetensors
