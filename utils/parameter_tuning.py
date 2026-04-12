import optuna
import numpy as np
import logging
from core.caoa_mod import CAOA
from core.jssp_env import JSSPSimulator
from core.encoder import ROVEncoder
from utils.tidal_builder import build_sparse_tidal_lookup

# Matumikan log Optuna agar terminal tidak terlalu kotor
optuna.logging.set_verbosity(optuna.logging.WARNING)

def main_tuning():
    print("=== INISIALISASI OPTUNA HYPERPARAMETER TUNING ===")
    voyage_path = 'data/processed/voyage_sim.csv'
    rules_path = 'data/raw/tidal_rules_0000.csv'
    tidal_data_path = 'data/raw/tidal_data.csv'

    global_tidal_lookup = build_sparse_tidal_lookup(rules_path, tidal_data_path)
    
    # Aturan Realitas
    max_voyage_rules = {
        # Kapal Single-Trayek
        "KM.KELUD": {"default": 168.0},
        "KM.BUKITRAYA": {"default": 336.0},
        "KM.LABOBAR": {"default": 336.0},
        "KM.GUNUNGDEMPO": {"default": 336.0},
        "KM.LAWIT": {"default": 336.0},
        "KM.TIDAR": {"default": 336.0},
        "KM.NGGAPULU": {"default": 336.0},
        "KM.CIREMAI": {"default": 336.0},
        "KM.SINABUNG": {"default": 336.0},
        "KM.AWU": {"default": 336.0},
        "KM.LEUSER": {"default": 672.0},
        "KM.EGON": {"default": 336.0},
        "KM.TILONGKABILA": {"default": 336.0},
        "KM.SIRIMAU": {"default": 672.0},
        "KM.WILIS": {"default": 336.0},
        "KM.LAMBELU": {"default": 336.0},
        "KM.BUKITSIGUNTANG": {"default": 336.0},
        "KFC.JETLINER": {"default": 168.0},
        "KM.TILONGKABILA": {"default": 336.0},
        "KM.SANGIANG": {"default": 336.0},
        "KM.PANGRANGO": {"default": 336.0},

        # Kapal Multi-Trayek
        "KM.DOBONSOLO": {
            "A": 384.0, 
            "B": 288.0
        },
        "KM.KELIMUTU": {
            "A": 288.0, 
            "B": 384.0
        },
        "KM.DOROLONDA": {
            "A": 336.0, 
            "B": 336.0
        },
        "KM.LAWIT": {
            "A": 336.0, 
            "B": 336.0
        },
        "KM.BINAIYA": {
            "A": 336.0, 
            "B": 336.0
        },
        "KM.TATAMAILAU": {
            "A": 336.0, 
            "B": 336.0
        },
    }
        
    encoder = ROVEncoder(voyage_path)
    simulator = JSSPSimulator(voyage_path, global_tidal_lookup, max_voyage_rules)
    dim = encoder.dim

    # =========================================================
    # KUNCI BOBOT BISNIS (JANGAN DIMASUKKAN KE SEARCH SPACE)
    # =========================================================
    OBJ_ALPHA = 1000.0  # Bobot Pelanggaran Tardiness
    OBJ_BETA  = 100.0   # Bobot Waktu Tunggu (Congestion & Tidal)
    OBJ_GAMMA = 10.0    # Bobot Inefisiensi Slack Publik
    # =========================================================

    def objective(trial):
        # 1. Search Space: Murni Mekanik Internal CAOA
        # (Diberi prefix 'c_' agar tidak tertukar dengan bobot objektif)
        c_alpha = trial.suggest_float('c_alpha', 0.05, 0.5) 
        c_beta = trial.suggest_float('c_beta', 0.01, 0.2)
        c_gamma_base = trial.suggest_float('c_gamma_base', 0.1, 3.0)
        c_delta = trial.suggest_int('c_delta', 1, 15)

        # Scaling metrik pencarian sesuai dimensi DDA-SLK (yang sekarang 2x lipat lebih besar)
        c_gamma_actual = c_gamma_base / np.sqrt(dim)
        
        # 2. Wrapper Objektif (Menggunakan fitur decode dimensi ganda)
        def fobj_wrapper(continuous_vector):
            legal_schedule, slack_map = encoder.decode(continuous_vector)
            # Hitung skor menggunakan bobot bisnis absolut yang dikunci
            fitness_score = simulator.evaluate_fitness(
                legal_schedule, slack_map, 
                alpha=OBJ_ALPHA, beta=OBJ_BETA, gamma=OBJ_GAMMA
            )
            return fitness_score

        # 3. Parameter Evaluasi Komputasi
        # Gunakan iterasi moderat (misal 150) agar proses tuning tidak memakan waktu berhari-hari. 
        # Tujuannya mencari parameter yang konvergen *lebih cepat*.
        N_pop = 50
        Max_Iter = 150 
        num_runs = 3
        results = []
        
        for _ in range(num_runs):
            # Perhatikan: lb=0.0 dan ub=1.0 sesuai batasan baru encoder DDA-SLK
            gBestScore, _, _ = CAOA(
                N=N_pop, max_iter=Max_Iter, 
                lb=0.0, ub=1.0, dim=dim, 
                fobj=fobj_wrapper,
                alpha=c_alpha, beta=c_beta, gamma=c_gamma_actual, delta=c_delta,
                initial_energy=100.0, verbose_interval=0
            )
            
            # Jika simulasi gagal keras (karena max loop dll), hukum dengan skor raksasa
            if gBestScore >= 1e9:
                return float('inf')
                
            results.append(gBestScore)
        
        # Meminimalkan rata-rata dari 3 run independen (Uji Ketangguhan/Robustness Parameter)
        return np.mean(results)

    # 4. Eksekusi Eksperimen Optuna
    # 4. Eksekusi Eksperimen Optuna
    print(f"Mencari Parameter CAOA Optimal untuk Ruang Pencarian {dim} Dimensi...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n" + "="*55)
    print("KOMBINASI PARAMETER MESIN TERBAIK (CAOA)")
    print("="*55)
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"{key:>15}: {value:.5f}")
        else:
            print(f"{key:>15}: {value}")
            
    # Kalkulasi dan tampilkan actual_gamma
    best_gamma_actual = study.best_params['c_gamma_base'] / np.sqrt(dim)
    print(f"{'c_gamma_actual':>15}: {best_gamma_actual:.5f} (Nilai Aktual yang dipakai CAOA)")
            
    print("-" * 55)
    print(f"Rata-rata Cost Minimum (Z) : {study.best_value:.2f}")
    print("="*55)
    
if __name__ == "__main__":
    main_tuning()