import pandas as pd

voyage_path = 'data/processed/voyage_sim.csv'

df = pd.read_csv(voyage_path)

# Agregasi data berdasarkan job_id, ship_name, dan voyage
agregasi = df.groupby(['job_id', 'ship_name', 'voyage'])[['travel_time', 'proc_time']].sum().reset_index()

# Konversi jam ke hari (1 hari = 24 jam)
agregasi['total_hari_layar'] = agregasi['travel_time'] / 24.0
agregasi['total_hari_labuh'] = agregasi['proc_time'] / 24.0

# Menghitung durasi 1 voyage
agregasi['durasi_1_voyage'] = agregasi['total_hari_layar'] + agregasi['total_hari_labuh']

# Menampilkan hasil
print(agregasi[['job_id', 'ship_name', 'voyage', 'total_hari_layar', 'total_hari_labuh', 'durasi_1_voyage']])