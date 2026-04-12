import numpy as np
import pandas as pd
from core.caoassr import CAOASSR
from core.encoder import ROVEncoder
from core.jssp_env import JSSPSimulator
from utils.tidal_builder import build_sparse_tidal_lookup

def main():
    print("=== INISIALISASI MESIN OPTIMASI (DDA-SLK ARCHITECTURE) ===")
    
    rules_path = 'data/raw/tidal_rules_0000.csv'
    tidal_data_path = 'data/raw/tidal_data.csv'
    voyage_path = 'data/processed/voyage_sim.csv'

    global_tidal_lookup = build_sparse_tidal_lookup(rules_path, tidal_data_path)
    
    # Aturan Realitas Operasional Anda (Sesuai Konteks Sebelumnya)
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
    
    print("\nMembangun Model Translasi Dimensional...")
    encoder = ROVEncoder(voyage_path)
    simulator = JSSPSimulator(voyage_path, global_tidal_lookup, max_voyage_rules)
    
    print(f"Total Operasi (N_w): {encoder.dim_single}")
    print(f"Dimensi Vektor CAOA (2 x N_w): {encoder.dim}")

    # Wrapper Objektif untuk CAOASSR
    def fobj_wrapper(continuous_vector):
        legal_schedule, slack_map = encoder.decode(continuous_vector)
        # Penalti Ekstrem pada Tardiness, Penalti Menengah pada Delay, Denda Ringan pada Slack
        return simulator.evaluate_fitness(legal_schedule, slack_map, alpha=1000.0, beta=100.0, gamma=10.0)

    # Parametrik CAOASSR 
    N_pop = 200
    max_iter = 1000
    
    print(f"\nMemulai Eksploitasi CAOASSR (Pop: {N_pop}, Iter: {max_iter})...")
    # lb dan ub = 0.0 hingga 1.0 (Batas absolut untuk Probabilitas Slack dan Rank-Order)
    best_Z, best_pos, convergence_curve = CAOASSR(
        N=N_pop, max_iter=max_iter, lb=0.0, ub=1.0, dim=encoder.dim, fobj=fobj_wrapper
    )
    
    print(f"\n[KONVERGENSI TERCAPAI] Cost Metrik Z Minimum: {best_Z:.2f}")
    
    # Ekstraksi Output Timetable yang Kebal Hukum Alam
    legal_schedule, slack_map = encoder.decode(best_pos)
    final_Z, timetable = simulator.evaluate_fitness(
        legal_schedule, slack_map, return_details=True
    )
    
    df_timetable = pd.DataFrame(timetable)
    output_path = "data/processed/optimized_timetable.csv"
    df_timetable.to_csv(output_path, index=False)
    print(f"Timetable Realistis Diekstrak Ke: '{output_path}'")
    
    print("\n" + "="*80)
    print("EVALUASI METRIK FISIK FINAL")
    print("="*80)
    print(f"Total Antrean Dermaga (W_cong)  : {df_timetable['W_cong'].sum():.1f} Jam")
    print(f"Total Interupsi Pasang (W_tidal): {df_timetable['W_tidal'].sum():.1f} Jam")
    print(f"Total Kelonggaran Publik (Slack): {df_timetable['slack_injected'].sum():.1f} Jam")
    print("="*80)

if __name__ == "__main__":
    main()