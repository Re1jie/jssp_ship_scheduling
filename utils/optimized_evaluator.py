import pandas as pd
import numpy as np

def evaluate_optimized_timetable(opt_csv_path, voyage_csv_path, max_voyage_rules):
    df_opt = pd.read_csv(opt_csv_path)
    df_voyage = pd.read_csv(voyage_csv_path)
    
    # Map Rute
    route_map = df_voyage.groupby('job_id')['rute'].first().to_dict()
    
    total_tardiness = 0.0
    
    print("\n" + "="*95)
    print("EVALUASI OPTIMIZED TIMETABLE (JOB-LEVEL)".center(95))
    print("="*95)
    print(f"{'Job ID':<8} | {'Ship Name':<15} | {'Rute':<6} | {'Start':<8} | {'Akhir':<8} | {'Durasi':<10} | {'Max D_i':<10} | {'Tardiness'}")
    print("-" * 95)
    
    for job_id, group in df_opt.groupby('job_id'):
        ship_name = group['ship_name'].iloc[0]
        rute = str(route_map.get(job_id, 'default'))
        if rute == 'nan' or not rute.strip(): rute = 'default'
        
        start_time = group['arrival_time'].min()
        end_time = group['C_ij'].max()  # Waktu penyelesaian fisik absolut
        durasi = end_time - start_time
        
        # Cari batas D_i
        d_i_max = max_voyage_rules.get(ship_name, {}).get(rute, max_voyage_rules.get(ship_name, {}).get("default", 336.0))
        
        tardiness = max(0, durasi - d_i_max)
        total_tardiness += tardiness
        
        print(f"{job_id:<8} | {ship_name:<15} | {rute:<6} | {start_time:<8.1f} | {end_time:<8.1f} | {durasi:<10.1f} | {d_i_max:<10.1f} | {tardiness:.1f}")

    print("-" * 95)
    print(f"Total Job-Level Tardiness : {total_tardiness:.1f} Jam")
    print(f"Total Kelonggaran Publik (Slack): {df_opt['slack_injected'].sum():.1f} Jam")
    print("="*95)

if __name__ == "__main__":
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
    
    evaluate_optimized_timetable(
        'data/processed/optimized_timetable.csv', 
        'data/processed/voyage_sim.csv', 
        max_voyage_rules
    )