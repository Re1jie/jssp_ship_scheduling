import pandas as pd
import heapq
from itertools import count

PORT_CAPACITY = {
    "TANJUNGPRIOK": 2,
    "MAKASSAR": 2,
    "BAU-BAU": 2,
    "AMBON": 2,
    "SORONG": 2,
    "KUPANG": 2
}

DEFAULT_CAPACITY = 1

def detect_berth_conflicts(df):
    conflicts = []
    df = df.copy()
    
    # Koreksi Logika: Waktu selesai aktual adalah waktu kedatangan ditambah waktu pemrosesan.
    # Asumsi: arrival_time dan proc_time memiliki format numerik atau datetime yang bisa dijumlahkan.
    # Jika tidak, kamu harus melakukan konversi tipe data terlebih dahulu sebelum baris ini.
    df["actual_finish"] = df["arrival_time"] + df["proc_time"]

    conflict_id_counter = 1

    for port, group in df.groupby("port_name"):
        capacity = PORT_CAPACITY.get(port, DEFAULT_CAPACITY)
        
        # Urutkan berdasarkan waktu kedatangan paling awal
        group = group.sort_values("arrival_time")

        active = []
        counter = count()  # tie breaker untuk min-heap

        for _, row in group.iterrows():
            start = row["arrival_time"]
            finish = row["actual_finish"]

            # Keluarkan kapal dari antrean aktif jika waktu selesainya <= waktu kedatangan kapal baru
            while active and active[0][0] <= start:
                heapq.heappop(active)

            # Masukkan kapal baru ke antrean aktif (disortir berdasarkan waktu selesai tercepat)
            heapq.heappush(active, (finish, next(counter), row))

            # Jika jumlah kapal di dermaga melebihi kapasitas, rekam semua kapal yang bersinggungan
            if len(active) > capacity:
                for _, _, conflict_row in active:
                    # Ambil nama kolom voyage (menyesuaikan jika namamu voyage_id atau voyage_code)
                    voyage_val = conflict_row.get("voyage", conflict_row.get("voyage_id", "N/A"))
                    
                    conflicts.append({
                        "conflict_id": f"CONF-{conflict_id_counter:04d}",
                        "port_name": port,
                        "port_capacity": capacity,
                        "conflict_trigger_time": start,
                        "voyage": voyage_val,
                        "ship_name": conflict_row["ship_name"],
                        "arrival_time": conflict_row["arrival_time"],
                        "proc_time": conflict_row["proc_time"],
                        "actual_finish": conflict_row["actual_finish"],
                        "due_date": conflict_row["due_date"]
                    })
                conflict_id_counter += 1

    conflict_df = pd.DataFrame(conflicts)
    
    # Hapus duplikasi jika diperlukan
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
        output_file = "berth_conflicts_report.csv"
        conflict_df.to_csv(output_file, index=False)
        
        print(f"Analisis Selesai. Ditemukan {conflict_df['conflict_id'].nunique()} insiden pelanggaran kapasitas.")
        print(f"Data diekspor ke: {output_file}\n")
        
        # Tampilkan sampel data untuk inspeksi terminal
        if not conflict_df.empty:
            print(conflict_df[["conflict_id", "port_name", "voyage", "ship_name", "arrival_time", "actual_finish"]].head(10))
        else:
            print("Tidak ada konflik jadwal yang terdeteksi.")
            
    except FileNotFoundError:
        print("Error: Pastikan path file benar.")