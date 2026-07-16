#include "bindings.hpp"

#ifdef QAQMC_USE_OPENMP
#include <omp.h>
#endif

PYBIND11_MODULE(qaqmc_cpp, m) {
    m.doc() = "CPU QAQMC and SSE engines with pybind11 bindings";

#ifdef QAQMC_USE_OPENMP
    m.attr("has_openmp") = true;
    m.attr("omp_max_threads") = omp_get_max_threads();
#else
    m.attr("has_openmp") = false;
    m.attr("omp_max_threads") = 1;
#endif

    bind_qaqmc(m);
    bind_qaqmc_renyi(m);
    bind_sse(m);
    bind_qaqmc_renyi_work(m);
}
