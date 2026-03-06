import time
import numpy as np
import tracemalloc
from src.qaqmc import QAQMC_Rydberg
from src.lattices import generate_kagome_bond_lattice

def run_profile():
    out = []
    out.append("="*70)
    out.append("QAQMC Internal C++ Profiling Report")
    out.append("="*70)
    
    pos = generate_kagome_bond_lattice(nx=2, ny=2, a=4.0)
    N = len(pos)
    M = 2000
    M_total = 2 * M
    n_bonds = N * 15 // 2
    OMP_THREADS = 4
    N_ITER = 100
    
    out.append(f"System: Kagome_bond (N={N}, bonds≈{n_bonds}), M={M}, M_total={M_total}")
    out.append(f"Threads: OMP_NUM_THREADS={OMP_THREADS}")
    out.append(f"Profiling steps: {N_ITER} MC steps per config")
    out.append("-" * 70)
    out.append(f"{'Chunk':>6} | {'Peak RAM':>9} | {'Total/MC':>9} | {'Diag/MC':>9} | {'Clus/MC':>9}")
    out.append(f"{'':>6} | {'(MB)':>9} | {'(ms)':>9} | {'(ms)':>9} | {'(ms)':>9}")
    out.append("-" * 70)

    chunk_list = [10, 50, 100, 500, 1000, 4000]
    
    for c in chunk_list:
        tracemalloc.start()
        
        q = QAQMC_Rydberg(
            N=N, M=M, Omega=1.0, Rb=2.4,
            delta_min=0.0, delta_max=8.0,
            epsilon=0.01, seed=42, omp_threads=OMP_THREADS,
            neighbor_cutoff=3, precompute=True,
            chunk_slices=c, pos=pos
        )
        q.init_kwargs['verbose'] = False # Disable tqdm
        
        q.mc_step()
        
        if q._cpp_engine is not None:
            q._cpp_engine.reset_timers()

        t0 = time.perf_counter()
        if q._cpp_engine is not None:
            try:
                q._cpp_engine.run(N_ITER, 0, None, 1)
            except TypeError:
                q._cpp_engine.run(N_ITER, 0)
        else:
            for _ in range(N_ITER):
                q.mc_step()
        total_time = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        steps = q._cpp_engine.mc_steps
        t_diag = q._cpp_engine.time_diag * 1000.0 / steps  # ms per step
        t_clus = q._cpp_engine.time_clus * 1000.0 / steps  # ms per step
        t_tot = total_time * 1000.0 / steps
        peak_mb = peak / (1024 * 1024)
        
        out.append(f"{c:6d} | {peak_mb:9.1f} | {t_tot:9.2f} | {t_diag:9.2f} | {t_clus:9.2f}")

    out.append("="*70)
    
    with open("profile_out.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    try:
        print("\n".join(out))
    except UnicodeEncodeError:
        pass

if __name__ == '__main__':
    run_profile()
