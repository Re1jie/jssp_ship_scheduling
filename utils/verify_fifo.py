import pandas as pd
from collections import defaultdict
from utils.tidal_builder import build_sparse_tidal_lookup

PORT_CAPACITY = {
    "TANJUNGPRIOK": 1,
    "MAKASSAR": 1,
    "BAUBAU": 1,
    "AMBON": 1,
    "SORONG": 1,
    "KUPANG": 1
}

def verify_fifo(voyage_csv, tidal_lookup):
    df = pd.read_csv(voyage_csv)
    df = df.sort_values(["arrival_time"])

    ship_time = defaultdict(float)
    port_time = {
        port: [0.0] * PORT_CAPACITY.get(port, 1)
        for port in df['port_name'].unique()
    }

    results = []
    total_tardiness = 0
    tidal_delays = 0
    port_conflicts = 0

    total_port_wait = 0
    total_ship_wait = 0
    port_wait_events = 0

    for _, row in df.iterrows():
        job = row.job_id
        ship = row.ship_name
        port = row.port_name

        arrival = row.arrival_time
        proc = row.proc_time
        buf = row.buffer_time
        due = row.due_date

        berths = port_time[port]
        berth_idx = min(range(len(berths)), key=lambda i: berths[i])
        berth_ready = berths[berth_idx]

        # waiting BEFORE scheduling
        port_wait = max(0, berth_ready - arrival)
        ship_wait = max(0, ship_time[job] - arrival)

        start = max(arrival, ship_time[job], berth_ready)
        original_start = start

        tidal_array = tidal_lookup.get(port, {}).get(ship, [])
        max_t = len(tidal_array)

        if max_t > 0:
            while True:
                if start + proc + buf >= max_t:
                    raise RuntimeError("Out of tidal range")

                if buf > 0:
                    entry_safe = all(
                        tidal_array[int(t)] 
                        for t in range(int(start - buf), int(start))
                        if t >= 0
                    )

                    if entry_safe:
                        finish = start + proc
                        depart = finish

                        while True:
                            exit_safe = all(
                                tidal_array[int(t)]
                                for t in range(int(depart), int(depart + buf))
                            )
                            if exit_safe:
                                break
                            depart += 1
                            tidal_delays += 1

                        finish = depart
                        break
                    else:
                        start += 1
                        tidal_delays += 1
                else:
                    docking_safe = all(
                        tidal_array[int(t)]
                        for t in range(int(start), int(start + proc))
                    )
                    if docking_safe:
                        finish = start + proc
                        break
                    else:
                        start += 1
                        tidal_delays += 1
        else:
            finish = start + proc

        # check port conflict
        if start < berth_ready:
            port_conflicts += 1

        total_port_wait += port_wait
        total_ship_wait += ship_wait

        if port_wait > 0:
            port_wait_events += 1

        ship_time[job] = finish
        port_time[port][berth_idx] = finish

        tardiness = max(0, finish - due)
        total_tardiness += tardiness

        results.append({
            "job": job,
            "port": port,
            "arrival": arrival,
            "start": start,
            "finish": finish,
            "due": due,
            "tardiness": tardiness,
            "port_wait": port_wait,
            "ship_wait": ship_wait
        })

    print("===== FIFO VERIFICATION =====")
    print("Total tardiness:", total_tardiness)
    print("Tidal delays:", tidal_delays)
    
    print("\n--- WAITING METRICS ---")
    print("Total port waiting time:", total_port_wait)
    print("Total ship waiting time:", total_ship_wait)
    print("Port waiting events:", port_wait_events)
    
    return pd.DataFrame(results)

if __name__ == "__main__":

    tidal_lookup = build_sparse_tidal_lookup(
        rules_csv_path="data/raw/tidal_rules.csv",
        tidal_csv_path="data/raw/tidal_data.csv"
    )

    df = verify_fifo(
        "data/processed/voyage_sim.csv",
        tidal_lookup
    )

    print(df[df.tardiness > 0])