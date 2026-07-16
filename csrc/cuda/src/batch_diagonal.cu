#include "include/diagonal.cuh"
#include "detail/diagonal_state.cuh"

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
void run_diagonal_chains_parallel(std::size_t count, Function&& function) {
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

struct BatchedDiagonalEngine::Impl {
    int batch_size{0};
    int n_sites{0};
    std::size_t length{0};
    std::vector<std::unique_ptr<DiagonalEngine>> chains;
};

BatchedDiagonalEngine::BatchedDiagonalEngine(
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
    batch.chains.reserve(static_cast<std::size_t>(batch_size));
    for (int chain = 0; chain < batch_size; ++chain) {
        const std::size_t offset = static_cast<std::size_t>(chain) * batch.length;
        batch.chains.push_back(std::make_unique<DiagonalEngine>(
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

BatchedDiagonalEngine::~BatchedDiagonalEngine() = default;
BatchedDiagonalEngine::BatchedDiagonalEngine(BatchedDiagonalEngine&&) noexcept = default;
BatchedDiagonalEngine& BatchedDiagonalEngine::operator=(
    BatchedDiagonalEngine&&) noexcept = default;

std::vector<DiagonalStats> BatchedDiagonalEngine::diagonal_update(
    const uint64_t* seeds, const uint64_t* sweep_ids) {
    std::vector<DiagonalStats> result(static_cast<std::size_t>(impl_->batch_size));
    run_diagonal_chains_parallel(result.size(), [&](std::size_t chain) {
        result[chain] = impl_->chains[chain]->diagonal_update(
            seeds[chain], sweep_ids[chain]);
    });
    return result;
}

std::vector<ClusterStats> BatchedDiagonalEngine::cluster_update(
    const uint64_t* seeds, const uint64_t* sweep_ids) {
    std::vector<ClusterStats> result(static_cast<std::size_t>(impl_->batch_size));
    run_diagonal_chains_parallel(result.size(), [&](std::size_t chain) {
        result[chain] = impl_->chains[chain]->cluster_update(
            seeds[chain], sweep_ids[chain]);
    });
    return result;
}

void BatchedDiagonalEngine::set_string_sites(
    const int32_t* sites, int count, int m_star) {
    run_diagonal_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->set_string_sites(sites, count, m_star);
    });
}

void BatchedDiagonalEngine::set_seam_masks_consistent(const uint64_t* masks) {
    run_diagonal_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->set_seam_mask_consistent(masks[chain]);
    });
}

std::vector<TopologyStats> BatchedDiagonalEngine::topology_sweep(
    double lambda, const uint64_t* seeds, const uint64_t* sweep_ids) {
    std::vector<TopologyStats> result(static_cast<std::size_t>(impl_->batch_size));
    run_diagonal_chains_parallel(result.size(), [&](std::size_t chain) {
        result[chain] = impl_->chains[chain]->topology_sweep(
            lambda, seeds[chain], sweep_ids[chain]);
    });
    return result;
}

void BatchedDiagonalEngine::save_checkpoint() {
    run_diagonal_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->save_checkpoint();
    });
}

void BatchedDiagonalEngine::restore_checkpoint() {
    run_diagonal_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->restore_checkpoint();
    });
}

bool BatchedDiagonalEngine::has_checkpoint() const {
    for (const auto& chain : impl_->chains) {
        if (!chain->has_checkpoint()) return false;
    }
    return true;
}

void BatchedDiagonalEngine::get_operator_strings(
    int32_t* types, int32_t* sites) const {
    run_diagonal_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        const std::size_t offset = chain * impl_->length;
        impl_->chains[chain]->get_operator_string(types + offset, sites + offset);
    });
}

void BatchedDiagonalEngine::set_operator_strings(
    const int32_t* types, const int32_t* sites) {
    run_diagonal_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        const std::size_t offset = chain * impl_->length;
        impl_->chains[chain]->set_operator_string(types + offset, sites + offset);
    });
}

void BatchedDiagonalEngine::get_profile_states(
    int profile_step, uint64_t* output) const {
    if (profile_step <= 0
        || static_cast<std::size_t>(profile_step) > impl_->length)
        throw std::invalid_argument(
            "profile_step must be in [1, operator length]");
    const std::size_t points = impl_->length / static_cast<std::size_t>(profile_step);
    const std::size_t words = static_cast<std::size_t>(
        impl_->chains.front()->packed_words());
    run_diagonal_chains_parallel(impl_->chains.size(), [&](std::size_t chain) {
        impl_->chains[chain]->get_profile_states(
            profile_step, output + chain * points * words);
    });
}

void BatchedDiagonalEngine::get_seam_masks(uint64_t* masks) const {
    for (std::size_t chain = 0; chain < impl_->chains.size(); ++chain)
        masks[chain] = impl_->chains[chain]->seam_mask();
}

int BatchedDiagonalEngine::batch_size() const { return impl_->batch_size; }
int BatchedDiagonalEngine::n_sites() const { return impl_->n_sites; }
std::size_t BatchedDiagonalEngine::length() const { return impl_->length; }
std::size_t BatchedDiagonalEngine::shared_model_bytes() const {
    return impl_->chains.front()->impl_->model->allocated_bytes();
}
std::size_t BatchedDiagonalEngine::device_bytes() const {
    std::size_t bytes = shared_model_bytes();
    for (const auto& chain : impl_->chains) bytes += chain->impl_->state_bytes();
    return bytes;
}


}  // namespace qaqmc_cuda
