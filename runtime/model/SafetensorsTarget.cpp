#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

namespace splash::model {

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)), experts_(experts) {
  planned_->images =
      safetensors::planImages(planned_->checkpoint, geometry, {.streamExperts = experts.memoryBytes != 0});
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
  return open(planned_->table);
}

WeightFile SafetensorsTargetLoader::mtp() {
  if (!planned_->mtp) throw WeightStoreError("the target holds no MTP head");
  return open(planned_->mtp);
}

} // namespace splash::model
