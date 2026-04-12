import pandas as pd
from collections import defaultdict
from utils.tidal_builder import build_sparse_tidal_lookup

PORT_CAPACITY = {
    "TANJUNGPRIOK": 2,
    "MAKASSAR": 2,
    "BAUBAU": 2,
    "AMBON": 2,
    "SORONG": 2,
    "KUPANG": 2
}

def verify_fifo_with_wait_isolation(voyage_csv, tidal_lookup):
    df = pd.read_csv(voyage_csv)
    # Urutkan berdasarkan kedatangan untuk simulasi FIFO sejati
    df = df.sort_values(["arrival_time"])

    ship_time = defaultdict(float)
    port_time = {
        port: [0.0] * PORT_CAPACITY.get(port, 1)
        for port in df['port_name'].unique()
    }

    results = []
    
    # Global Metrics terisolasi
    global_tardiness = 0
    global_congestion_wait = 0
    global_tidal_wait = 0

    for _, row in df.iterrows():
        job = row.job_id
        ship = row.ship_name
        port = row.port_name

        arrival = row.arrival_time
        proc = row.proc_time
        buf = row.buffer_time
        due = row.due_date

        # 1. Kapan kapal SECARA FISIK siap di area pelabuhan?
        # Harus memperhitungkan jika kapal telat dari rute sebelumnya
        ready_to_dock = max(arrival, ship_time[job]) 

        # 2. Kapan dermaga pelabuhan kosong?
        berths = port_time[port]
        berth_idx = min(range(len(berths)), key=lambda i: berths[i])
        berth_ready = berths[berth_idx]

        # --- ISOLASI 1: CONGESTION WAIT ---
        # Kapal sudah siap, tapi harus menunggu dermaga kosong
        congestion_wait = max(0, berth_ready - ready_to_dock)
        
        # Waktu paling awal kapal bisa masuk jika mengabaikan alam
        theoretical_start = max(ready_to_dock, berth_ready)
        actual_start = theoretical_start

        tidal_array = tidal_lookup.get(port, {}).get(ship, [])
        max_t = len(tidal_array)

        # Evaluasi Pasang Surut
        if max_t > 0:
            while True:
                if actual_start + proc + buf >= max_t:
                    raise RuntimeError(f"Out of tidal range for {ship} at {port}")

                if buf > 0:
                    # Cek keamanan masuk (buffer sebelum proses)
                    entry_safe = all(
                        tidal_array[int(t)] 
                        for t in range(int(actual_start - buf), int(actual_start))
                        if t >= 0
                    )

                    if entry_safe:
                        finish = actual_start + proc
                        depart = finish

                        # Cek keamanan keluar (buffer setelah proses)
                        while True:
                            exit_safe = all(
                                tidal_array[int(t)]
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
                    # Tanpa buffer, cek keamanan selama proses docking
                    docking_safe = all(
                        tidal_array[int(t)]
                        for t in range(int(actual_start), int(actual_start + proc))
                    )
                    if docking_safe:
                        finish = actual_start + proc
                        break
                    else:
                        actual_start += 1
        else:
            finish = actual_start + proc

        # --- ISOLASI 2: TIDAL WAIT ---
        # Kapal sudah siap, dermaga sudah kosong, tapi harus menunggu air pasang
        tidal_wait = actual_start - theoretical_start

        # Update Tracker
        ship_time[job] = finish
        port_time[port][berth_idx] = finish

        tardiness = max(0, finish - due)
        
        # Akumulasi Global
        global_tardiness += tardiness
        global_congestion_wait += congestion_wait
        global_tidal_wait += tidal_wait

        results.append({
            "job": job,
            "ship": ship,
            "port": port,
            "arrival_time": arrival,
            "theoretical_start": theoretical_start,
            "actual_start": actual_start,
            "finish": finish,
            "due_date": due,
            "congestion_wait": congestion_wait,
            "tidal_wait": tidal_wait,
            "tardiness": tardiness
        })

    print("===== METRIK ISOLASI JSSP-TW (FIFO) =====")
    print(f"Total Tardiness Akhir : {global_tardiness} Jam")
    print(f"Total Waktu Hilang akibat Antrean Pelabuhan (Congestion) : {global_congestion_wait} Jam")
    print(f"Total Waktu Hilang akibat Pasang Surut Alam (Tidal Wait) : {global_tidal_wait} Jam")
    print("=========================================")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    tidal_lookup = build_sparse_tidal_lookup(
        rules_csv_path="data/raw/tidal_rules_0000.csv",
        tidal_csv_path="data/raw/tidal_data.csv"
    )

    df_results = verify_fifo_with_wait_isolation(
        "data/processed/voyage_sim.csv",
        tidal_lookup
    )

    # Tampilkan kapal yang menjadi korban pasang surut secara spesifik
    print("\n[Detail Operasi Korban Pasang Surut]")
    print(df_results[df_results.tidal_wait > 0][['ship', 'port', 'congestion_wait', 'tidal_wait', 'tardiness']])