#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::model {
namespace {

// A streamed table's rows from the checkpoint's shards (the image's copies,
// each a shard's rows in order): row r of the table is the row the image
// would hold at r, built from its shard's tensors (readCopyRows).
class TableRowReader final : public RowReader {
public:
  TableRowReader(std::shared_ptr<const SafetensorsCheckpoint> checkpoint, const gguf::Image &image,
                 uint32_t rowBytes)
      : checkpoint_(std::move(checkpoint)), rowBytes_(rowBytes) {
    for (const gguf::Copy &copy : image.copies) {
      if (copy.conversion != gguf::Conversion::None || copy.source.rowBytes != rowBytes)
        throw std::invalid_argument("a streamed table's shards are its native rows as stored");
      parts_.push_back({rows_, copy.source});
      rows_ += copy.source.rows;
    }
  }
  void read(uint64_t offset, std::span<uint8_t> bytes) override {
    if (offset % rowBytes_ || bytes.size() % rowBytes_ || offset / rowBytes_ + bytes.size() / rowBytes_ > rows_)
      throw SparseRowError("a streamed table's read is not whole rows of it");
    for (uint64_t row = offset / rowBytes_, done = 0; done < bytes.size(); ++row, done += rowBytes_) {
      const auto part = std::upper_bound(parts_.begin(), parts_.end(), row,
                                         [](uint64_t value, const Part &p) { return value < p.first; }) - 1;
      readCopyRows(part->rows, row - part->first, 1, bytes.data() + done);
    }
  }
  void checkUnchanged() override {
    try {
      checkpoint_->checkUnchanged();
    } catch (const std::exception &error) {
      throw SparseRowError(error.what());
    }
  }
  [[nodiscard]] uint64_t size() const override { return rows_ * rowBytes_; }

private:
  struct Part final {
    uint64_t first = 0;
    gguf::TensorRows rows;
  };
  std::shared_ptr<const SafetensorsCheckpoint> checkpoint_;
  uint32_t rowBytes_ = 0;
  uint64_t rows_ = 0;
  std::vector<Part> parts_;
};

} // namespace

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, ExpertStoreOptions experts,
                                                 SparseRowStoreOptions table)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)), experts_(experts) {
  planned_->images = safetensors::planImages(
      planned_->checkpoint, geometry,
      {.streamExperts = experts.memoryBytes != 0, .streamTable = geometry.hasPle() && table.cacheBytes != 0});
  planned_->tableStore = std::move(table);
  planned_->layers = geometry.layers;
  size_t next = geometry.layers + 2;
  if (geometry.hasPle()) planned_->table = next++;
  if (safetensors::hasMtp(planned_->checkpoint, geometry)) planned_->mtp = next++;
  if (planned_->images.size() != next) throw WeightStoreError("the target's images do not match its plan");
  float32Norms_ = safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint), geometry);
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  return {gguf::streamedExperts(std::span(planned_->images).first(planned_->layers)), planned_,
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkpoint.checkUnchanged();
                       },
                       plan.residentBytes});
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) {
  if (index >= planned_->layers) throw WeightStoreError("target layer is out of range");
  return open(index);
}

WeightFile SafetensorsTargetLoader::head() { return open(planned_->layers); }

WeightFile SafetensorsTargetLoader::embedding() { return open(planned_->layers + 1); }

WeightFile SafetensorsTargetLoader::table() {
  if (!planned_->table) throw WeightStoreError("the target holds no table");
  if (planned_->images[planned_->table].streamedRows) throw WeightStoreError("the target's table streams");
  return open(planned_->table);
}

std::optional<GgufTargetLoader::StreamedTable> SafetensorsTargetLoader::streamTable() const {
  if (!planned_->table) return std::nullopt;
  const gguf::Image &plan = planned_->images[planned_->table];
  if (!plan.streamedRows) return std::nullopt;
  const gguf::StreamedRows &rows = *plan.streamedRows;
  planned_->checkpoint.checkUnchanged();
  auto reader = std::make_unique<TableRowReader>(
      std::shared_ptr<const SafetensorsCheckpoint>(planned_, &planned_->checkpoint), plan, rows.rowBytes);
  return GgufTargetLoader::StreamedTable{
      std::make_shared<SparseRowStore>(SparseRowTable{rows.tensor, rows.rows, rows.rowBytes, 0}, std::move(reader),
                                       planned_->tableStore),
      rows.format,
      // As WeightImages::load records the image (WeightFile).
      {"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes, images_.contentIdentity()}};
}

WeightFile SafetensorsTargetLoader::mtp() {
  if (!planned_->mtp) throw WeightStoreError("the target holds no MTP head");
  return open(planned_->mtp);
}

} // namespace splash::model
