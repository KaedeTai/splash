#include "Qwen4Exp.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/ExpertStore.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/SparseRowStore.hpp"

#include <cmath>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace splash::model {
namespace {

// The PLE metadata's integers, which the parser keeps as doubles
// (GgufFile::numericArray), are below these: a multiplier below 2^53, past
// which a double does not hold every integer, and a head's offset or row
// count below 2^31, a 32-bit signed value.
constexpr double kMultiplierLimit = 0x1p53, kHeadValueLimit = 0x1p31;

// The sections of one mix, in the image's order (GgufImage.cpp hyperImage).
Qwen4HyperWeights readHyper(WeightFile &file, const Qwen4ExpLayout &layout, bool inject, const std::string &label) {
  const BlockTargetFormat format;
  const uint32_t streams = layout.streamWidth();
  Qwen4HyperWeights hyper;
  hyper.norm = format.norm(file, streams, label + "-norm");
  hyper.down = format.projection(file, layout.hyperRank, streams, label + "-down");
  hyper.up = format.projection(file, streams, layout.hyperRank, label + "-up");
  if (inject)
    hyper.inject = file.section(uint64_t{layout.hyperConnections} * streams * kFloat32Bytes, label + "-inject");
  return hyper;
}

Qwen4IndexerWeights readIndexer(WeightFile &file, const Qwen4ExpLayout &layout) {
  const BlockTargetFormat format;
  Qwen4IndexerWeights indexer;
  indexer.query = format.projection(file, layout.indexerHeads * layout.indexerHeadDimension, layout.hiddenSize,
                                    "indexer-query");
  indexer.key = format.projection(file, layout.indexerHeadDimension, layout.hiddenSize, "indexer-key");
  // The indexer scores in fp32 (ops::BlockIndexer).
  indexer.query.destination = ops::FloatOutput::Float32;
  indexer.key.destination = ops::FloatOutput::Float32;
  indexer.queryNorm = format.norm(file, layout.indexerHeadDimension, "indexer-query-norm");
  indexer.keyNorm = format.norm(file, layout.indexerHeadDimension, "indexer-key-norm");
  return indexer;
}

// Throws unless Flash-Next's own dimensions are set: the hyper-connections,
// the QSA indexer, and a PLE n-gram embedding of 2 to kPleMaximumNgram tokens
// that feeds a GDN layer. loadQwenTarget checks the hybrid's
// (requireQwenLayout).
void requireQwen4Layout(const Qwen4ExpLayout &layout) {
  if (!layout.hyperConnections || !layout.hyperRank || !layout.indexerHeads || !layout.indexerHeadDimension ||
      !layout.indexerTopBlocks || !layout.indexerBlockTokens || layout.pleNgram < 2 ||
      layout.pleNgram > kPleMaximumNgram || !layout.pleHeadsPerNgram || !layout.pleHeadDimension ||
      !layout.pleConvolutionTaps || layout.pleLayer >= layout.layers || layout.isFullAttentionLayer(layout.pleLayer))
    throw WeightStoreError("Qwen3.8-Flash-Next layout is inconsistent");
}

// The PLE hash the GGUF's metadata states: every multiplier the array holds a whole
// number, the first pleNgram the hash's; each head's rows inside the table
// (per_layer_token_embd), whose rows are 32-bit ids; and its end-of-sequence
// token and the image token it may name the layout's.
Qwen4PleHash readPleHash(const GgufFile &file, const Qwen4ExpLayout &layout) {
  const std::string key = std::string(Qwen4ExpLayout::ggufArchitecture) + ".ple.";
  const auto array = [&](const char *name, size_t count) {
    const std::optional<std::span<const double>> values = file.numericArray(key + name);
    if (!values || values->size() < count) throw GgufError("GGUF PLE metadata is missing: ple." + std::string(name));
    return *values;
  };
  const auto whole = [](double value, double limit) {
    if (!(value >= 0 && value < limit) || value != std::floor(value))
      throw GgufError("GGUF PLE metadata holds an invalid value");
    return value;
  };
  Qwen4PleHash hash;
  const std::span<const double> multipliers = array("layer_multipliers", layout.pleNgram);
  for (size_t position = 0; position < multipliers.size(); ++position) {
    const auto multiplier = static_cast<uint64_t>(whole(multipliers[position], kMultiplierLimit));
    if (position < layout.pleNgram) hash.multipliers[position] = multiplier;
  }
  const uint32_t heads = layout.pleHeads();
  const std::span<const double> offsets = array("head_offsets", heads), sizes = array("head_vocab_sizes", heads);
  hash.tableRows = file.require("per_layer_token_embd.weight").rows();
  for (uint32_t head = 0; head < heads; ++head) {
    hash.headOffsets.push_back(static_cast<uint32_t>(whole(offsets[head], kHeadValueLimit)));
    hash.headVocabularies.push_back(static_cast<uint32_t>(whole(sizes[head], kHeadValueLimit)));
    if (!hash.headVocabularies.back() ||
        uint64_t{hash.headOffsets.back()} + hash.headVocabularies.back() > hash.tableRows)
      throw GgufError("a PLE head's rows lie outside per_layer_token_embd");
  }
  const std::optional<uint64_t> eos = file.metadata().unsignedValue(key + "eos_token_id");
  const std::optional<uint64_t> image = file.metadata().unsignedValue(key + "image_token_id");
  if (!eos) throw GgufError("GGUF PLE metadata is missing: ple.eos_token_id");
  if (*eos != layout.pleEosToken || (image && *image != layout.pleImageToken) || hash.tableRows > UINT32_MAX)
    throw WeightStoreError("the GGUF's PLE hash does not match the layout");
  return hash;
}

// The PLE hash an MLX checkpoint's tensors state (mlx-vlm's
// Qwen4ExpNGramEmbedding): its I64 layer_multipliers, every one a whole
// number below kMultiplierLimit, the first pleNgram the hash's; each head's
// rows inside the table (ngram_heads_offsets, ngram_heads_vocab_sizes),
// whose rows are those of its shards (safetensors::pleShards). Its config
// states the end-of-sequence and image tokens, which the configuration check
// held to the layout's (model/ModelDescriptor.mm).
Qwen4PleHash readPleHash(const SafetensorsCheckpoint &checkpoint, const Qwen4ExpLayout &layout) {
  const safetensors::ModuleNames names = safetensors::moduleNames(checkpoint);
  const std::string module = safetensors::pleEmbeddingModule(layout, names);
  const auto integers = [&](const char *name, size_t count) {
    const SourceTensor &tensor = checkpoint.require(module + name);
    if (tensor.dtype != "I64" || tensor.shape.size() != 1 || tensor.shape[0] < count ||
        tensor.bytes != tensor.shape[0] * sizeof(int64_t))
      throw WeightStoreError("PLE tensor type or shape does not match: " + module + name);
    std::vector<int64_t> values(tensor.shape[0]);
    tensor.read(0, {reinterpret_cast<uint8_t *>(values.data()), tensor.bytes});
    return values;
  };
  const auto whole = [](int64_t value, double limit) {
    if (value < 0 || static_cast<double>(value) >= limit) throw WeightStoreError("PLE tensor holds an invalid value");
    return static_cast<uint64_t>(value);
  };
  Qwen4PleHash hash;
  const std::vector<int64_t> multipliers = integers("layer_multipliers", layout.pleNgram);
  for (size_t position = 0; position < multipliers.size(); ++position) {
    const uint64_t multiplier = whole(multipliers[position], kMultiplierLimit);
    if (position < layout.pleNgram) hash.multipliers[position] = multiplier;
  }
  for (const safetensors::PleShard &shard : safetensors::pleShards(checkpoint, layout, names))
    hash.tableRows += shard.rows;
  const uint32_t heads = layout.pleHeads();
  const std::vector<int64_t> offsets = integers("ngram_heads_offsets", heads),
                             sizes = integers("ngram_heads_vocab_sizes", heads);
  for (uint32_t head = 0; head < heads; ++head) {
    hash.headOffsets.push_back(static_cast<uint32_t>(whole(offsets[head], kHeadValueLimit)));
    hash.headVocabularies.push_back(static_cast<uint32_t>(whole(sizes[head], kHeadValueLimit)));
    if (!hash.headVocabularies.back() ||
        uint64_t{hash.headOffsets.back()} + hash.headVocabularies.back() > hash.tableRows)
      throw WeightStoreError("a PLE head's rows lie outside its table");
  }
  if (hash.tableRows > UINT32_MAX) throw WeightStoreError("the PLE table holds more rows than 32-bit ids reach");
  return hash;
}

// The PLE hash of a target's source: its GGUF's metadata, or its checkpoint's
// tensors.
Qwen4PleHash readPleHash(GgufTargetLoader &gguf, const Qwen4ExpLayout &layout) {
  return readPleHash(gguf.file(), layout);
}
Qwen4PleHash readPleHash(SafetensorsTargetLoader &safetensors, const Qwen4ExpLayout &layout) {
  return readPleHash(safetensors.checkpoint(), layout);
}

} // namespace

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                    const QwenTargetFiles &files) {
  requireQwen4Layout(layout);
  return loadQwenTarget<Qwen4ExpWeights>(
      backend, layout, files, [&](auto &source, const BlockTargetFormat &format, const MoeExperts &experts) {
        // A GGUF's images or a safetensors checkpoint's (model/SafetensorsImage.hpp),
        // which hold the same sections in the same order.
        constexpr bool fromGguf = std::is_same_v<std::decay_t<decltype(source)>, GgufTargetLoader>;
        Qwen4ExpWeights result;
        result.layout = layout;
        // The PLE hash first, from the GGUF's metadata or the checkpoint's
        // tensors, before any image loads.
        result.pleHash = readPleHash(source, layout);
        result.layers.reserve(layout.layers);
        const uint32_t streams = layout.streamWidth();
        for (uint32_t index = 0; index < layout.layers; ++index) {
          const bool fullAttention = layout.isFullAttentionLayer(index);
          WeightFile file = source.layer(index);
          Qwen4ExpLayerWeights &layer = result.layers.emplace_back();
          layer.mixerHyper = readHyper(file, layout, true, "mixer-hyper");
          layer.mixer = readQwenMixer(file, format, layout, fullAttention);
          if (fullAttention) layer.indexer = readIndexer(file, layout);
          layer.ffnHyper = readHyper(file, layout, true, "ffn-hyper");
          layer.ffn = readMoeBlock(file, experts, index);
          if (index == layout.pleLayer) {
            Qwen4PleWeights ple;
            ple.key = format.projection(file, streams, layout.pleWidth(), "ple-key");
            ple.value = format.projection(file, layout.hiddenSize, layout.pleWidth(), "ple-value");
            ple.keyNorm = format.norm(file, streams, "ple-key-norm");
            ple.queryNorm = format.norm(file, streams, "ple-query-norm");
            ple.convolutionNorm = format.norm(file, streams, "ple-convolution-norm");
            ple.convolution =
                file.section(uint64_t{streams} * layout.pleConvolutionTaps * kFloat32Bytes, "ple-convolution");
            layer.ple = std::move(ple);
          }
          file.finish();
          result.files.push_back(file.record());
        }
        {
          WeightFile file = source.head();
          result.finalHyper = readHyper(file, layout, false, "final-hyper");
          result.logitsProjection = format.projection(file, layout.vocabularySize, layout.hiddenSize, "logits");
          result.logitsProjection.destination = ops::FloatOutput::Float32;
          file.finish();
          result.files.push_back(file.record());
        }
        {
          WeightFile file = source.embedding();
          result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
          file.finish();
          result.files.push_back(file.record());
        }
        if constexpr (fromGguf) {
          if (std::optional<GgufTargetLoader::StreamedTable> table = source.streamTable()) {
            if (table->store->table().rows != result.pleHash.tableRows)
              throw WeightStoreError("the GGUF's PLE table does not match its hash");
            result.streamedTable = std::move(table->store);
            result.pleFormat = table->format;
            result.files.push_back(std::move(table->record));
            return result;
          }
        }
        {
          WeightFile file = source.table();
          result.pleTable = readBlockEmbedding(file, static_cast<uint32_t>(result.pleHash.tableRows),
                                               layout.pleHeadDimension, "ple-table");
          file.finish();
          result.files.push_back(file.record());
        }
        if constexpr (!fromGguf) {
          if (source.hasMtp()) {
            // The image's sections in their order (SafetensorsImage.cpp walkMtp);
            // its experts are resident whether the target's stream or not.
            WeightFile file = source.mtp();
            Qwen4MtpWeights mtp;
            mtp.embeddingNorm = format.norm(file, layout.hiddenSize, "mtp-embedding-norm");
            mtp.hiddenNorm = format.norm(file, streams, "mtp-hidden-norm");
            mtp.embeddingProjection =
                format.projection(file, layout.hiddenSize, layout.hiddenSize, "mtp-embedding-projection");
            mtp.hiddenProjection = format.projection(file, layout.hiddenSize, layout.hiddenSize, "mtp-hidden-projection");
            mtp.mixerHyper = readHyper(file, layout, true, "mtp-mixer-hyper");
            mtp.attention = std::get<QwenAttentionWeights>(readQwenMixer(file, format, layout, true));
            mtp.ffnHyper = readHyper(file, layout, true, "mtp-ffn-hyper");
            MoeExperts resident;
            resident.resident = residentExpertSlots(backend, layout.experts);
            resident.experts = layout.experts;
            mtp.ffn = readMoeBlock(file, resident, layout.layers);
            mtp.finalHyper = readHyper(file, layout, false, "mtp-final-hyper");
            mtp.head = format.projection(file, layout.vocabularySize, layout.hiddenSize, "mtp-head");
            mtp.head.destination = ops::FloatOutput::Float32;
            file.finish();
            result.files.push_back(file.record());
            result.mtp = std::move(mtp);
          }
        }
        return result;
      });
}

} // namespace splash::model
