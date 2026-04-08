import pandas as pd
import numpy as np

class JSSPSimulator:
    def __init__(self, voyage_csv_path, global_tidal_lookup):
        # Dari tidal_builder
        self.global_tidal_lookup = global_tidal_lookup
        
        # Variabel ini akan diisi saat _build_voyage_lookup berjalan
        self.unique_ports = set()

         # Mapping job_id ke ship_name
        self.ship_names = {}
        
        # Bangun Lookup Table 
        self.voyage_lookup = self._build_voyage_lookup(voyage_csv_path)

        # Kapasitas pelabuhan
        PORT_CAPACITY = {
            "TANJUNGPRIOK": 2,
            "MAKASSAR": 2,
            "BAUBAU": 2,
            "AMBON": 2,
            "SORONG": 2,
            "KUPANG": 2
        }

        self.port_capacity = {
            port: PORT_CAPACITY.get(port, 1)
            for port in self.unique_ports
        }

        # Ekstrak daftar job_id untuk inisialisasi tracker nanti
        self.job_ids = list(self.voyage_lookup.keys())

    def _build_voyage_lookup(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=['job_id', 'op_seq'])
        
        lookup = {}
        for job_id, group in df.groupby('job_id'):
            self.ship_names[job_id] = group['ship_name'].iloc[0]
            
            ops_data = []
            for _, row in group.iterrows():
                self.unique_ports.add(row['port_name'])
                ops_data.append({
                    'port_name': row['port_name'],
                    'm_id': row['m_id'],
                    'initial_arrival': row['arrival_time'],
                    'travel_time_to_next': row['travel_time'],
                    'proc_time': row['proc_time'],
                    'buffer_time': row['buffer_time'],
                    'due_date': row['due_date']
                })
            lookup[job_id] = ops_data
            
        return lookup
    
    def evaluate_fitness(self, legal_schedule):
        # Inisialisasi State Trackers
        op_tracker = {job_id: 0 for job_id in self.job_ids}
        ship_time_tracker = {job_id: 0.0 for job_id in self.job_ids}
        port_time_tracker = {
            port: [0.0] * self.port_capacity[port]
            for port in self.unique_ports
        }

        total_tardiness = 0.0
        
        # Iterasi Prioritas Jadwal (Membaca array dari kiri ke kanan)
        for job_id in legal_schedule:
            current_op_idx = op_tracker[job_id]
            op_details = self.voyage_lookup[job_id][current_op_idx]
            
            port_name = op_details['port_name']
            ship_name = self.ship_names[job_id]
            proc_t = op_details['proc_time']
            buf_t = op_details['buffer_time']
            due_date = op_details['due_date']

            # Fase 1 & 2: Kedatangan Aktual & Waktu Siap Pelabuhan
            if current_op_idx == 0:
                # Jika ini pelabuhan pertama, gunakan kedatangan historis
                arrival_time = op_details['initial_arrival']
            else:
                # Jika ini pelabuhan ke-2 dst, tarik waktu tempuh dari pelabuhan sebelumnya
                prev_op_details = self.voyage_lookup[job_id][current_op_idx - 1]
                travel_from_prev = prev_op_details['travel_time_to_next']
                arrival_time = ship_time_tracker[job_id] + travel_from_prev
                
            berths = port_time_tracker[port_name]

            # pilih berth paling cepat tersedia
            berth_idx = min(range(len(berths)), key=lambda i: berths[i])
            berth_ready = berths[berth_idx]

            tentative_start = max(arrival_time, berth_ready)

            actual_start = int(tentative_start)
            proc = int(proc_t)
            buf = int(buf_t)

            # Ambil array pasang surut spesifik
            tidal_dict = self.global_tidal_lookup.get(port_name, {})
            tidal_array = tidal_dict.get(ship_name, [])

            # Batas akhir kalender pasang surut
            max_t = len(tidal_array)

            if max_t == 0:
                # Jika tidak ada kalender pasang surut, kapal langsung bersandar
                actual_finish = actual_start + proc
            else:
                # Fase 3: Pencarian Jendela Pasang Surut
                while True:
                    # Jika waktu pencarian melebihi data kalender pasang surut
                    if actual_start + proc + buf >= max_t:
                        # Kembalikan penalti absolut. Algoritma harus menjauhi jadwal buruk ini.
                        # return float('inf') 
                        return 1e10

                    if buf > 0:
                        # ATURAN 1 (Pelabuhan dengan Alur Masuk/Keluar)
                        start_check = max(0, actual_start - buf)
                        entry_safe = all(tidal_array[t] for t in range(start_check, actual_start))
                        
                        if entry_safe:
                            tentative_finish = actual_start + proc
                            actual_departure = tentative_finish
                            
                            while True:
                                # Proteksi out of bounds saat mencari jadwal keluar
                                if actual_departure + buf >= max_t:
                                    # return float('inf')
                                    return 1e10
                                
                                exit_safe = all(tidal_array[t] for t in range(actual_departure, actual_departure + buf))
                                if exit_safe:
                                    break 
                                actual_departure += 1 # Kapal nyangkut di pelabuhan menunggu surut
                                
                            actual_finish = actual_departure
                            break 
                            
                        else:
                            actual_start += 1 # Kapal tertahan di laut (delay kedatangan)
                            
                    else:
                        # ATURAN 2 (Pelabuhan Dangkal / Tanpa Alur)
                        docking_safe = all(tidal_array[t] for t in range(actual_start, actual_start + proc))
                        
                        if docking_safe:
                            actual_finish = actual_start + proc
                            break
                        else:
                            actual_start += 1 

            # Fase 4: Eksekusi dan Pembaruan Masa Depan (State Update)
            ship_time_tracker[job_id] = actual_finish
            port_time_tracker[port_name][berth_idx] = actual_finish
            op_tracker[job_id] += 1

            op_tardiness = max(0, actual_finish - due_date)

            # Hitung Objektif
            total_tardiness += op_tardiness
            
        # Makespan: Waktu di mana kapal terakhir menyelesaikan tugasnya
        # makespan = max(ship_time_tracker.values())
        # composite_fitness = total_tardiness + (0.00001 * makespan)
        return total_tardiness