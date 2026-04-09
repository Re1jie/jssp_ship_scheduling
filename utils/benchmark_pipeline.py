import numpy as np
import pandas as pd
import time
from scipy import stats

# Import kedua algoritma
from core.caoa_mod import CAOA
from core.caoassr import CAOASSR
from core.encoder import ROVEncoder
from core.jssp_env import JSSPSimulator
from utils.tidal_builder import build_sparse_tidal_lookup

def run_benchmark(num_runs=30, N=150, max_iter=1000, lb=-1.0, ub=1.0):
    print("=== INISIALISASI ENVIRONMENT JSSP (HANYA SEKALI) ===")
    
    rules_path = 'data/raw/tidal_rules.csv'
    tidal_data_path = 'data/raw/tidal_data.csv'
    voyage_path = 'data/processed/voyage_sim.csv'

    global_tidal_lookup = build_sparse_tidal_lookup(rules_path, tidal_data_path)
    encoder = ROVEncoder(voyage_path)
    simulator = JSSPSimulator(voyage_path, global_tidal_lookup)
    dim = encoder.dim
    
    def fobj_wrapper(continuous_vector):
        legal_schedule = encoder.decode(continuous_vector)
        return simulator.evaluate_fitness(legal_schedule)

    print(f"Dimensi Vektor: {dim}")
    print(f"Memulai Benchmark: {num_runs} Runs | N={N} | Iter={max_iter}\n")

    # Penyimpanan hasil
    results = []

    for run in range(1, num_runs + 1):
        print(f"--- Eksekusi Run {run}/{num_runs} ---")
        
        # KONTROL SEED: Pastikan kedua algoritma mendapatkan "keberuntungan" awal yang sama
        # Jika algoritma Anda menggunakan np.random di dalamnya, menetapkan seed di sini
        # membantu mereplikasi kondisi awal untuk eksperimen yang adil.
        np.random.seed(run)
        
        # --- Eksekusi CAOA Standar ---
        start_time = time.time()
        best_fit_caoa, _, _ = CAOA(
            N=N, max_iter=max_iter, lb=lb, ub=ub, dim=dim, 
            fobj=fobj_wrapper, verbose_interval=0 # Matikan verbose agar terminal bersih
        )
        time_caoa = time.time() - start_time
        
        # --- Eksekusi CAOA + SSR ---
        # Reset seed agar sequence random sama persis dengan yang dialami CAOA
        np.random.seed(run)
        
        start_time = time.time()
        best_fit_ssr, _, _ = CAOASSR(
            N=N, max_iter=max_iter, lb=lb, ub=ub, dim=dim, 
            fobj=fobj_wrapper, verbose_interval=0
        )
        time_ssr = time.time() - start_time

        print(f"  CAOA     -> Fitness: {best_fit_caoa:.4f} | Waktu: {time_caoa:.2f}s")
        print(f"  CAOA+SSR -> Fitness: {best_fit_ssr:.4f} | Waktu: {time_ssr:.2f}s")
        
        results.append({
            'Run': run,
            'CAOA_Fitness': best_fit_caoa,
            'CAOA_Time': time_caoa,
            'SSR_Fitness': best_fit_ssr,
            'SSR_Time': time_ssr
        })

    # --- ANALISIS STATISTIK ---
    print("\n=== HASIL ANALISIS STATISTIK ===")
    df_results = pd.DataFrame(results)
    
    stats_summary = pd.DataFrame({
        'Metric': ['Mean (Rata-rata)', 'Std Dev (Stabilitas)', 'Min (Terbaik)', 'Max (Terburuk)'],
        'CAOA_Fitness': [
            df_results['CAOA_Fitness'].mean(),
            df_results['CAOA_Fitness'].std(),
            df_results['CAOA_Fitness'].min(),
            df_results['CAOA_Fitness'].max()
        ],
        'CAOA+SSR_Fitness': [
            df_results['SSR_Fitness'].mean(),
            df_results['SSR_Fitness'].std(),
            df_results['SSR_Fitness'].min(),
            df_results['SSR_Fitness'].max()
        ],
        'CAOA_Time (s)': [
            df_results['CAOA_Time'].mean(),
            df_results['CAOA_Time'].std(),
            df_results['CAOA_Time'].min(),
            df_results['CAOA_Time'].max()
        ],
        'CAOA+SSR_Time (s)': [
            df_results['SSR_Time'].mean(),
            df_results['SSR_Time'].std(),
            df_results['SSR_Time'].min(),
            df_results['SSR_Time'].max()
        ]
    })
    
    print(stats_summary.to_string(index=False))

    # Uji Signifikansi (Wilcoxon Signed-Rank Test)
    # Digunakan karena hasil metaheuristik jarang berdistribusi normal sempurna
    stat, p_value = stats.wilcoxon(df_results['CAOA_Fitness'], df_results['SSR_Fitness'])
    
    print("\n=== UJI SIGNIFIKANSI (Wilcoxon Test) ===")
    print(f"P-Value: {p_value:.5e}")
    if p_value < 0.05:
        print("Kesimpulan: Perbedaan performa antara CAOA dan CAOA+SSR adalah SIGNIFIKAN secara statistik (p < 0.05).")
        if df_results['SSR_Fitness'].mean() < df_results['CAOA_Fitness'].mean():
            print("Status: CAOA+SSR TERBUKTI lebih superior dalam meminimalkan Tardiness.")
        else:
            print("Status: CAOA standar secara mengejutkan lebih baik. Strategi SSR Anda mungkin cacat.")
    else:
        print("Kesimpulan: TIDAK ADA perbedaan yang signifikan (p >= 0.05).")
        print("Status: Modifikasi SSR Anda tidak memberikan dampak nyata atau hanya membuang-buang komputasi.")

    # Simpan hasil mentah untuk plotting/analisis lanjutan
    df_results.to_csv('benchmark_results.csv', index=False)
    print("\nHasil mentah disimpan ke 'benchmark_results.csv'.")

if __name__ == "__main__":
    # Paksa 30 run untuk validitas empiris. 
    # Jika komputasi terlalu berat, perkecil N atau max_iter, tapi JANGAN kurangi num_runs di bawah 30.
    run_benchmark(num_runs=30, N=150, max_iter=1000)