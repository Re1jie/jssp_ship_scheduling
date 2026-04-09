import optuna
import numpy as np
from core.caoa_mod import CAOA
from core.jssp_env import JSSPSimulator
from core.encoder import ROVEncoder
from utils.tidal_builder import build_sparse_tidal_lookup

voyage_path = 'data/processed/voyage_sim.csv'
rules_path = 'data/raw/tidal_rules.csv'
tidal_data_path = 'data/raw/tidal_data.csv'

global_tidal_lookup = build_sparse_tidal_lookup(rules_path, tidal_data_path)
    
encoder = ROVEncoder(voyage_path)
simulator = JSSPSimulator(voyage_path, global_tidal_lookup)
dim = encoder.dim

def fobj_wrapper(continuous_vector):
        legal_schedule = encoder.decode(continuous_vector)
        fitness_score = simulator.evaluate_fitness(legal_schedule)
        return fitness_score

def objective(trial):
    # Search Space
    alpha = trial.suggest_float('alpha', 0.1, 0.9)
    beta = trial.suggest_float('beta', 0.01, 0.1)
    gamma_base = trial.suggest_float('gamma_base', 0.1, 2.0)
    delta = trial.suggest_int('delta', 1, 10)

    # Scaling Gamma
    gamma_actual = gamma_base / np.sqrt(dim)
    
    # Parameter Simulasi Tuning
    N_pop = 50
    Max_Iter = 300 
    num_runs = 3
    results = []
    
    for _ in range(num_runs):
        # Panggil CAOA yang telah distabilkan, gunakan fobj_wrapper!
        gBestScore, _, _ = CAOA(
            N=N_pop, max_iter=Max_Iter, 
            lb=-1.0, ub=1.0, dim=dim, 
            fobj=fobj_wrapper,
            alpha=alpha, beta=beta, gamma=gamma_actual, delta=delta,
            initial_energy=10.0, verbose_interval=100 
        )
        results.append(gBestScore)
    
    # Meminimalkan rata-rata dari 3 run independen
    return np.mean(results)

# 5. Eksekusi Ekperimen
if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("\n" + "="*50)
    print("Kombinasi Parameter Terbaik Ditemukan:")
    print("="*50)
    for key, value in study.best_params.items():
        print(f"{key:>12}: {value}")
    print(f"Rata-rata Tardiness Terbaik: {study.best_value:.2f} jam")