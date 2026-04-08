import pandas as pd
import heapq
from itertools import count

PORT_CAPACITY = {
    "TANJUNGPRIOK": 2,
    "MAKASSAR": 2,
    "BAUBAU": 2,
    "AMBON": 2,
    "SORONG": 2,
    "KUPANG": 2
}

DEFAULT_CAPACITY = 1


def detect_berth_conflicts(df):
    conflicts = []
    df = df.copy()

    # finish time assuming direct berth
    df["actual_finish"] = df["arrival_time"] + df["proc_time"]

    conflict_id = 1

    for port, group in df.groupby("port_name"):

        capacity = PORT_CAPACITY.get(port, DEFAULT_CAPACITY)

        group = group.sort_values("arrival_time")

        # heap: (finish_time, counter, row_index)
        active = []
        counter = count()

        for idx, row in group.iterrows():

            arrival = row["arrival_time"]
            finish = row["actual_finish"]

            # remove finished ships
            while active and active[0][0] <= arrival:
                heapq.heappop(active)

            # add new ship
            heapq.heappush(active, (finish, next(counter), idx))

            # check multi-berth overflow
            if len(active) > capacity:

                for f, _, conflict_idx in active:
                    r = df.loc[conflict_idx]

                    conflicts.append({
                        "conflict_id": f"CONF-{conflict_id:05d}",
                        "port_name": port,
                        "capacity": capacity,
                        "trigger_time": arrival,
                        "job_id": r.get("job_id"),
                        "ship_name": r.get("ship_name"),
                        "voyage": r.get("voyage"),
                        "arrival_time": r["arrival_time"],
                        "finish_time": r["actual_finish"],
                        "proc_time": r["proc_time"],
                        "due_date": r["due_date"]
                    })

                conflict_id += 1

    conflict_df = pd.DataFrame(conflicts)

    if not conflict_df.empty:
        conflict_df = conflict_df.drop_duplicates()

    return conflict_df

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/voyage_dummy.csv")
        
        # Validasi eksistensi kolom sebelum eksekusi
        required_cols = ["port_name", "ship_name", "arrival_time", "proc_time", "due_date"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset kehilangan kolom wajib: {missing_cols}")

        conflict_df = detect_berth_conflicts(df)
        
        # Ekspor ke CSV agar dapat diobservasi
        output_file = "multiberth_conflicts_report.csv"
        conflict_df.to_csv(output_file, index=False)
        
        print(f"Analisis Selesai. Ditemukan {conflict_df['conflict_id'].nunique()} insiden pelanggaran kapasitas.")
        print(f"Data diekspor ke: {output_file}\n")
        
        # Tampilkan sampel data untuk inspeksi terminal
        if not conflict_df.empty:
            print(conflict_df[
                ["conflict_id", "port_name", "voyage", "ship_name", "arrival_time", "finish_time"]
            ].head(10))
        else:
            print("Tidak ada konflik jadwal yang terdeteksi.")
            
    except FileNotFoundError:
        print("Error: Pastikan path file benar.")