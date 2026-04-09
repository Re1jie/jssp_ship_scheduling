import numpy as np
import pandas as pd
import time
# from core.caoa import CAOA
from core.caoa_mod import CAOA
from core.encoder import ROVEncoder
from core.jssp_env import JSSPSimulator
from utils.tidal_builder import build_sparse_tidal_lookup

# def get_fifo_schedule(voyage_csv):
#     df = pd.read_csv(voyage_csv)
#     # Urutkan berdasarkan waktu kedatangan aktual
#     df = df.sort_values(["arrival_time"])
#     # Kembalikan sebagai list job_id berurutan
#     return df['job_id'].tolist()

def main():
    print("=== INISIALISASI ENVIRONMENT JSSP ===")
    
    rules_path = 'data/raw/tidal_rules.csv'
    tidal_data_path = 'data/raw/tidal_data.csv'
    voyage_path = 'data/processed/voyage_sim.csv'

    global_tidal_lookup = build_sparse_tidal_lookup(rules_path, tidal_data_path)
    
    print("\nMembangun Encoder dan Pre-computing Simulator...")
    encoder = ROVEncoder(voyage_path)
    simulator = JSSPSimulator(voyage_path, global_tidal_lookup)
    
    dim = encoder.dim
    print(f"Dimensi Vektor CAOA (Total Operasi): {dim}")

    # print("\nMengekstrak Jadwal FIFO sebagai Seed Target...")
    # fifo_schedule = get_fifo_schedule(voyage_path)
    # seed_vector = encoder.encode(fifo_schedule)

    def fobj_wrapper(continuous_vector):
        legal_schedule = encoder.decode(continuous_vector)
        fitness_score = simulator.evaluate_fitness(legal_schedule)
        return fitness_score

    # Konfigurasi Parameter
    N = 150
    max_iter = 1000
    lb = -1.0
    ub = 1.0
    
    print(f"\nMemulai Evolusi CAOA (N={N}, Iter={max_iter}, D={dim})...")
    start_time = time.time()

    # Masukkan benih yang sudah dieksekusi ke dalam parameter
    best_fitness, best_position, convergence_curve = CAOA(
        N=N, 
        max_iter=max_iter, 
        lb=lb, 
        ub=ub, 
        dim=dim,
        fobj=fobj_wrapper,
        verbose_interval=10,
        # seed_position=seed_vector
    )
    
    end_time = time.time()
    print(f"\nEvolusi Selesai dalam {(end_time - start_time):.2f} detik.")
    print(f"Fitness Terbaik (Tardiness Penalty): {best_fitness}")

if __name__ == "__main__":
    main()