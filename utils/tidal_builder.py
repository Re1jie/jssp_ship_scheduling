import pandas as pd
import numpy as np

def build_sparse_tidal_lookup(
    rules_csv_path,
    tidal_csv_path,
    anchor_date_str='2025-01-01 00:00:00'
):
    """
    Build sparse tidal lookup dari single source-of-truth CSV.
    """

    print("Membangun Tidal Lookup Table...")

    # --- Load rules
    rules_df = pd.read_csv(rules_csv_path)
    anchor_date = pd.to_datetime(anchor_date_str)

    # --- Load tidal (single source)
    tidal_df = pd.read_csv(tidal_csv_path)

    ts = tidal_df['timestamp'].astype(str)

    mask_24 = ts.str.contains('24:00:00')
    ts_fixed = ts.str.replace('24:00:00', '00:00:00', regex=False)

    tidal_df['datetime'] = pd.to_datetime(ts_fixed, errors='coerce')

    # 24:00 -> copy ke 00:00 hari berikutnya
    tidal_df.loc[mask_24, 'datetime'] += pd.Timedelta(days=1)

    tidal_df = tidal_df.dropna(subset=['datetime'])

    # --- Build time index
    tidal_df['t'] = (
        (tidal_df['datetime'] - anchor_date)
        .dt.total_seconds()
        .floordiv(3600)
        .astype(int)
    )

    tidal_df = tidal_df[tidal_df['t'] >= 0]

    global_tidal_lookup = {}

    # --- group once by port
    for port, tidal_port in tidal_df.groupby('port_name'):

        max_t = tidal_port['t'].max()
        elevations = np.full(max_t + 1, np.nan)

        elevations[tidal_port['t'].values] = tidal_port['tidal_elevation'].values

        # FIX boundary awal: jika t=0 kosong, isi dari nilai valid pertama
        if np.isnan(elevations[0]):
            first_valid = np.flatnonzero(~np.isnan(elevations))
            if len(first_valid) > 0:
                elevations[0] = elevations[first_valid[0]]

        global_tidal_lookup[port] = {}

        rules_port = rules_df[rules_df['port_name'] == port]

        for _, row in rules_port.iterrows():
            ship = row['ship_name']
            e_min = row['E_min']
            e_max = row['E_max']

            global_tidal_lookup[port][ship] = (
                (elevations >= e_min) &
                (elevations <= e_max)
            )

    print("Tidal Lookup Table berhasil dibangun!")
    return global_tidal_lookup

# DEBUGGING
def export_tidal_lookup_to_csv(global_tidal_lookup, output_csv_path):
    """
    Flatten global_tidal_lookup menjadi CSV agar dapat diobservasi.
    """

    print(f"Mengekspor tidal lookup ke: {output_csv_path}")

    rows = []

    for port, ships in global_tidal_lookup.items():
        for ship, valid_array in ships.items():
            for t, val in enumerate(valid_array):
                rows.append({
                    "port_name": port,
                    "ship_name": ship,
                    "t": t,
                    "is_allowed": bool(val)
                })

    df = pd.DataFrame(rows)

    # sorting biar deterministic
    df = df.sort_values(["port_name", "ship_name", "t"]).reset_index(drop=True)

    df.to_csv(output_csv_path, index=False)

    print("Export selesai.")

if __name__ == "__main__":
    lookup = build_sparse_tidal_lookup(
        'data/raw/tidal_rules.csv',
        'data/raw/tidal_data.csv'
    )

    export_tidal_lookup_to_csv(
        lookup,
        "tidal_lookup_debug.csv"
    )