// Compile-only gate for the legacy C++ include paths. Keeping this translation
// unit in the CPU module prevents compatibility headers from silently rotting.
#include "diagonal_observables.hpp"
#include "qaqmc_core.hpp"
#include "qaqmc_off_diagonal_core.hpp"
#include "qaqmc_renyi_core.hpp"
#include "qaqmc_renyi_work_core.hpp"
#include "sse_core.hpp"
#include "sse_off_diagonal_core.hpp"
