import pandas as pd
from collections import defaultdict
from utils.tidal_builder import build_sparse_tidal_lookup

PORT_CAPACITY = {
    "TANJUNGPRIOK": 2, "MAKASSAR": 2, "BAUBAU": 2,
    "AMBON": 2, "SORONG": 2, "KUPANG": 2, "LEMBAR": 1 # Tambahkan jika ada port lain
}

def evaluate_baseline_historical(voyage_csv, tidal_lookup, max_voyage_dict=None):
    df = pd.read_csv(voyage_csv)
    df = df.sort_values(["arrival_time"])

    ship_time = defaultdict(float)
    port_time = {
        port: [0.0] * PORT_CAPACITY.get(port, 1)
        for port in df['port_name'].unique()
    }

    results = []
    
    # Global Metrics
    global_congestion_wait = 0
    global_tidal_wait = 0
    total_negative_slack = 0 # Menghitung total jam di mana jadwal publik meleset dari realitas fisik

    for _, row in df.iterrows():
        job = row.job_id
        ship = row.ship_name
        port = row.port_name
        op_seq = row.op_seq

        rute = row.get('rute', 'default')
        if pd.isna(rute) or str(rute).strip() == "":
            rute = "default"

        arrival = row.arrival_time
        proc = row.proc_time
        buf = row.buffer_time
        due = row.due_date # Historis Due Date

        ready_to_dock = max(arrival, ship_time[job]) 

        berths = port_time[port]
        berth_idx = min(range(len(berths)), key=lambda i: berths[i])
        berth_ready = berths[berth_idx]

        # 1. Congestion Wait
        congestion_wait = max(0, berth_ready - ready_to_dock)
        theoretical_start = max(ready_to_dock, berth_ready)
        actual_start = theoretical_start

        tidal_array = tidal_lookup.get(port, {}).get(ship, [])
        max_t = len(tidal_array)

        # 2. Tidal Wait (Dengan Siklus Modulo untuk mencegah crash)
        if max_t > 0:
            while True:
                if buf > 0:
                    entry_safe = all(
                        tidal_array[int(t) % max_t] 
                        for t in range(int(actual_start - buf), int(actual_start))
                        if t >= 0
                    )
                    if entry_safe:
                        finish = actual_start + proc
                        depart = finish
                        while True:
                            exit_safe = all(
                                tidal_array[int(t) % max_t]
                                for t in range(int(depart), int(depart + buf))
                            )
                            if exit_safe:
                                break
                            depart += 1
                        finish = depart
                        break
                    else:
                        actual_start += 1
                else:
                    docking_safe = all(
                        tidal_array[int(t) % max_t]
                        for t in range(int(actual_start), int(actual_start + proc))
                    )
                    if docking_safe:
                        finish = actual_start + proc
                        break
                    else:
                        actual_start += 1
        else:
            finish = actual_start + proc

        tidal_wait = actual_start - theoretical_start

        ship_time[job] = finish
        port_time[port][berth_idx] = finish

        # 3. Implied Slack (Kelonggaran Jadwal Historis vs Fisik)
        # Jika finish > due, implied_slack negatif (Jadwal Mustahil/Telat)
        implied_slack = due - finish
        if implied_slack < 0:
            total_negative_slack += abs(implied_slack)

        global_congestion_wait += congestion_wait
        global_tidal_wait += tidal_wait

        results.append({
            "job": job,
            "ship": ship,
            "rute": rute,
            "op_seq": op_seq,
            "port": port,
            "arrival_time": arrival,
            "actual_start": actual_start,
            "finish": finish,
            "historical_due_date": due,
            "congestion_wait": congestion_wait,
            "tidal_wait": tidal_wait,
            "implied_slack": implied_slack
        })

    df_results = pd.DataFrame(results)

    # 4. Job-Level Tardiness Evaluation
    print("\n" + "="*95)
    print("EVALUASI BASELINE HISTORIS (JOB-LEVEL)".center(95))
    print("="*95)
    
    global_job_tardiness = 0
    
    first_ops = df_results.loc[df_results.groupby('job')['op_seq'].idxmin()].set_index('job')
    last_ops = df_results.loc[df_results.groupby('job')['op_seq'].idxmax()].set_index('job')
    
    print(f"{'Job ID':<8} | {'Ship Name':<15} | {'Rute':<6} | {'Start':<8} | {'Akhir':<8} | {'Durasi':<10} | {'Max D_i':<10} | {'Tardiness'}")
    print("-" * 95)
    
    for job in last_ops.index:
        ship = last_ops.loc[job, 'ship']
        rute = last_ops.loc[job, 'rute'] # Ambil rute dari dataframe hasil
        
        waktu_mulai_voyage = first_ops.loc[job, 'arrival_time']
        c_i_akhir = last_ops.loc[job, 'finish']
        durasi_aktual = c_i_akhir - waktu_mulai_voyage
        
        # --- LOGIKA MULTI-TRAYEK ---
        d_i_max = None
        if max_voyage_dict and ship in max_voyage_dict:
            aturan_kapal = max_voyage_dict[ship]
            if isinstance(aturan_kapal, dict):
                # Cari batas waktu spesifik rute, jika tidak ada, gunakan 'default'
                d_i_max = aturan_kapal.get(rute, aturan_kapal.get("default", None))
            else:
                # Fallback ke format lama jika kamu lupa mengubahnya jadi dictionary
                d_i_max = aturan_kapal
                
        if d_i_max is None:
            # Fallback pamungkas jika kapal tidak terdaftar di aturan bisnis sama sekali
            due_akhir_historis = last_ops.loc[job, 'historical_due_date']
            d_i_max = due_akhir_historis - waktu_mulai_voyage
            
        # Hitung Tardiness
        tardiness = max(0, durasi_aktual - d_i_max)
        global_job_tardiness += tardiness
        
        print(f"{job:<8} | {ship:<15} | {rute:<6} | {waktu_mulai_voyage:<8.1f} | {c_i_akhir:<8.1f} | {durasi_aktual:<10.1f} | {d_i_max:<10.1f} | {tardiness:.1f}")

    print("-" * 95)
    print(f"Total Job-Level Tardiness : {global_job_tardiness:.1f} Jam")
    print(f"Total Congestion Wait     : {global_congestion_wait:.1f} Jam")
    print(f"Total Tidal Wait          : {global_tidal_wait:.1f} Jam")
    print(f"Total Defisit Waktu (Negative Slack): {total_negative_slack:.1f} Jam")
    print("="*95)
    
    return df_results

if __name__ == "__main__":
    # Gunakan Dictionary ini untuk menguji Hard Constraint Bisnis (Contoh: Max 14 hari / 336 jam)
    # Ini bisa disesuaikan dengan data riil dari PELNI
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

    tidal_lookup = build_sparse_tidal_lookup(
        rules_csv_path="data/raw/tidal_rules_0000.csv",
        tidal_csv_path="data/raw/tidal_data.csv"
    )

    df_baseline = evaluate_baseline_historical(
        "data/processed/voyage_sim.csv",
        tidal_lookup,
        max_voyage_dict=max_voyage_rules
    )