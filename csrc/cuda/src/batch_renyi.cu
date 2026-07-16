#include "include/renyi.cuh"
#include "detail/renyi_state.cuh"

#include <cuda_runtime.h>

#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace qaqmc_cuda {
using namespace detail;
namespace {

template <typename Function>
void run_renyi_chains_parallel(std::size_t count, Function&& function) {
    if (count == 1) {
        function(0);
        return;
    }
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> failures(count);
    workers.reserve(count);
    for (std::size_t chain = 0; chain < count; ++chain) {
        workers.emplace_back([&, chain]() {
            try {
                function(chain);
            } catch (...) {
                failures[chain] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& failure : failures) {
        if (failure) std::rethrow_exception(failure);
    }
}

}  // namespace

struct BatchedRenyiEngine::Impl {
    int batch_size{0};
    int n_sites{0};
    std::size_t length{0};
    std::vector<std::unique_ptr<RenyiEngine>> chains;
};

BatchedRenyiEngine::BatchedRenyiEngine(
    int batch_size,
    int n_sites,
    int half_length,
    double delta_min,
    double delta_max,
    double epsilon,
    int n_groups,
    int max_alias,
    int n_bonds,
    const int32_t* bond_sites,
    const double* bond_vij,
    const double* inv_coord,
    const double* alias_prob,
    const int32_t* alias_index,
    const int32_t* alias_loc_kind,
    const double* bond_rmax,
    const int32_t* op_types,
    const int32_t* op_sites,
    int device_index)
    : impl_(std::make_unique<Impl>()) {
    if (batch_size <= 0)
        throw std::invalid_argument("batch_size must be positive");
    Impl& batch = *impl_;
    batch.batch_size = batch_size;
    batch.n_sites = n_sites;
    batch.length = static_cast<std::size_t>(2) * half_length;
    const std::size_t chain_stride = 2 * batch.length;
    batch.chains.reserve(static_cast<std::size_t>(batch_size));
    for (int chain = 0; chain < batch_size; ++chain) {
        const std::size_t offset = static_cast<std::size_t>(chain) * chain_stride;
        batch.chains.push_back(std::make_unique<RenyiEngine>(
            n_sites, half_length, delta_min, delta_max, epsilon,
            n_groups, max_alias, n_bonds, bond_sites, bond_vij, inv_coord,
            alias_prob, alias_index, alias_loc_kind, bond_rmax,
            op_types + offset, op_sites + offset, device_index));
        if (chain > 0) {
            auto& state = *batch.chains.back()->impl_;
            const auto& model = batch.chains.front()->impl_->model;
            state.model = model;
            state.bond_sites = DeviceBuffer<int32_t>::view(
                model->bond_sites.get(), model->bond_sites.size());
            state.bond_vij = DeviceBuffer<double>::view(
                model->bond_vij.get(), model->bond_vij.size());
            state.inv_coord = DeviceBuffer<double>::view(
                model->inv_coord.get(), model->inv_coord.size());
            state.alias_prob = DeviceBuffer<double>::view(
                model->alias_prob.get(), model->alias_prob.size());
            state.alias_index = DeviceBuffer<int32_t>::view(
                model->alias_index.get(), model->alias_index.size());
            state.alias_loc_kind = DeviceBuffer<int32_t>::view(
                model->alias_loc_kind.get(), model->alias_loc_kind.size());
            state.bond_rmax = DeviceBuffer<double>::view(
                model->bond_rmax.get(), model->bond_rmax.size());
        }
    }
}

BatchedRenyiEngine::~BatchedRenyiEngine() = default;
BatchedRenyiEngine::BatchedRenyiEngine(BatchedRenyiEngine&&) noexcept = default;
BatchedRenyiEngine& BatchedRenyiEngine::operator=(
    BatchedRenyiEngine&&) noexcept = default;

void BatchedRenyiEngine::set_cut(int cut) {
    for (auto& chain : impl_->chains) chain->set_cut(cut);
}

void BatchedRenyiEngine::set_masks(const uint8_t* masks) {
    run_renyi_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->set_mask(
            masks + chain * static_cast<std::size_t>(impl_->n_sites),
            impl_->n_sites);
    });
}

void BatchedRenyiEngine::get_masks(uint8_t* masks) const {
    run_renyi_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->get_mask(
            masks + chain * static_cast<std::size_t>(impl_->n_sites),
            impl_->n_sites);
    });
}

std::vector<DiagonalStats> BatchedRenyiEngine::diagonal_update(
    const uint64_t* seeds, const uint64_t* sweep_ids) {
    std::vector<DiagonalStats> result(static_cast<std::size_t>(impl_->batch_size));
    run_renyi_chains_parallel(result.size(), [&](std::size_t chain) {
        result[chain] = impl_->chains[chain]->diagonal_update(
            seeds[chain], sweep_ids[chain]);
    });
    return result;
}

std::vector<ClusterStats> BatchedRenyiEngine::cluster_update(
    const uint64_t* seeds, const uint64_t* sweep_ids) {
    std::vector<ClusterStats> result(static_cast<std::size_t>(impl_->batch_size));
    run_renyi_chains_parallel(result.size(), [&](std::size_t chain) {
        result[chain] = impl_->chains[chain]->cluster_update(
            seeds[chain], sweep_ids[chain]);
    });
    return result;
}

std::vector<TopologyStats> BatchedRenyiEngine::topology_sweep(
    const int32_t* topology_sites, int count, double lambda,
    const uint64_t* seeds, const uint64_t* sweep_ids) {
    std::vector<TopologyStats> result(static_cast<std::size_t>(impl_->batch_size));
    run_renyi_chains_parallel(result.size(), [&](std::size_t chain) {
        result[chain] = impl_->chains[chain]->topology_sweep(
            topology_sites, count, lambda, seeds[chain], sweep_ids[chain]);
    });
    return result;
}

void BatchedRenyiEngine::save_checkpoint() {
    run_renyi_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->save_checkpoint();
    });
}

void BatchedRenyiEngine::restore_checkpoint() {
    run_renyi_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->restore_checkpoint();
    });
}

bool BatchedRenyiEngine::has_checkpoint() const {
    for (const auto& chain : impl_->chains) {
        if (!chain->has_checkpoint()) return false;
    }
    return true;
}

void BatchedRenyiEngine::get_operator_strings(
    int32_t* types, int32_t* sites) const {
    const std::size_t stride = 2 * impl_->length;
    run_renyi_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->get_operator_strings(
            types + chain * stride, sites + chain * stride);
    });
}

void BatchedRenyiEngine::set_operator_strings(
    const int32_t* types, const int32_t* sites) {
    const std::size_t stride = 2 * impl_->length;
    run_renyi_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->set_operator_strings(
            types + chain * stride, sites + chain * stride);
    });
}

int BatchedRenyiEngine::batch_size() const { return impl_->batch_size; }
int BatchedRenyiEngine::n_sites() const { return impl_->n_sites; }
std::size_t BatchedRenyiEngine::length() const { return impl_->length; }
std::size_t BatchedRenyiEngine::shared_model_bytes() const {
    return impl_->chains.front()->impl_->model->allocated_bytes();
}
std::size_t BatchedRenyiEngine::device_bytes() const {
    std::size_t bytes = shared_model_bytes();
    for (const auto& chain : impl_->chains) bytes += chain->impl_->state_bytes();
    return bytes;
}


}  // namespace qaqmc_cuda
