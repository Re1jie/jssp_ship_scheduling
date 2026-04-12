import pandas as pd
import numpy as np

class JSSPSimulator:
    def __init__(self, voyage_csv_path, global_tidal_lookup, max_voyage_rules=None):
        self.global_tidal_lookup = global_tidal_lookup
        self.max_voyage_rules = max_voyage_rules or {}
        
        self.unique_ports = set()
        self.ship_names = {}
        self.ship_routes = {}
        
        self.voyage_lookup = self._build_voyage_lookup(voyage_csv_path)
        
        PORT_CAPACITY = {
            "TANJUNGPRIOK": 2, "MAKASSAR": 2, "BAUBAU": 2,
            "AMBON": 2, "SORONG": 2, "KUPANG": 2, "LEMBAR": 1
        }
        self.port_capacity = {port: PORT_CAPACITY.get(port, 1) for port in self.unique_ports}
        self.job_ids = list(self.voyage_lookup.keys())
        
        # Evaluator Keras: T_pure vs D_i
        self.T_pure = {}
        self.D_star = {}
        self._calculate_physical_bounds()

    def _build_voyage_lookup(self, csv_path):
        df = pd.read_csv(csv_path).sort_values(by=['job_id', 'op_seq'])
        lookup = {}
        for job_id, group in df.groupby('job_id'):
            ops = []
            for i in range(len(group)):
                row = group.iloc[i]
                self.unique_ports.add(row['port_name'])
                self.ship_names[job_id] = row['ship_name']
                self.ship_routes[job_id] = row.get('rute', 'default')
                
                # Derivasi Waktu Tempuh ke Pelabuhan Selanjutnya
                tt_next = 0.0
                if i < len(group) - 1:
                    next_row = group.iloc[i+1]
                    if 'travel_time' in row:
                        tt_next = row['travel_time']
                    else:
                        # Asumsi fallback: Selisih kedatangan dikurangi proc_time pelabuhan saat ini
                        tt_next = max(0.0, next_row['arrival_time'] - (row['arrival_time'] + row['proc_time']))

                ops.append({
                    'port_name': row['port_name'],
                    'proc_time': row['proc_time'],
                    'buffer_time': row.get('buffer_time', 0),
                    'initial_arrival': row['arrival_time'],
                    'tt_next': tt_next
                })
            lookup[job_id] = ops
        return lookup

    def _calculate_physical_bounds(self):
        for job_id, ops in self.voyage_lookup.items():
            # Waktu selesai teoritis paling cepat berdasarkan Start historis
            start_time = ops[0]['initial_arrival']
            total_proc = sum(op['proc_time'] for op in ops)
            total_tt = sum(op['tt_next'] for op in ops)
            
            # Asumsi kapal tiba di port terakhir, selesai proc, tanpa slack/antrean
            earliest_possible_finish = start_time + total_proc + total_tt 
            t_pure_effective = earliest_possible_finish - start_time
            
            self.T_pure[job_id] = t_pure_effective
            
            ship_name = self.ship_names[job_id]
            rute = self.ship_routes[job_id]
            
            d_rule = self.max_voyage_rules.get(ship_name, {}).get(rute, 
                    self.max_voyage_rules.get(ship_name, {}).get("default", 336.0))
            
            # Bandingkan batas SK Trayek dengan batas historis efektif
            # Kita juga bisa memasukkan durasi baseline historis jika Anda ingin lebih aman
            self.D_star[job_id] = max(d_rule, t_pure_effective)

    def evaluate_fitness(self, legal_schedule, slack_map, alpha=1000.0, beta=100.0, gamma=5.0, return_details=False):
        op_tracker = {job_id: 0 for job_id in self.job_ids}
        ship_time_tracker = {job_id: 0.0 for job_id in self.job_ids}
        due_date_tracker = {job_id: 0.0 for job_id in self.job_ids}
        ship_first_arrival = {job_id: 0.0 for job_id in self.job_ids}
        
        port_time_tracker = {port: [0.0] * self.port_capacity[port] for port in self.unique_ports}

        global_W_cong, global_W_tidal, global_q = 0.0, 0.0, 0.0
        timetable = []

        for job_id in legal_schedule:
            current_idx = op_tracker[job_id]
            op_details = self.voyage_lookup[job_id][current_idx]
            
            port_name = op_details['port_name']
            proc = int(op_details['proc_time'])
            buf = int(op_details['buffer_time'])
            tt_next = op_details['tt_next']

            # 1. State Passing (Kedatangan bergantung pada jadwal Due Date di pelabuhan sebelumnya)
            if current_idx == 0:
                arrival_time = op_details['initial_arrival']
                ship_first_arrival[job_id] = arrival_time
            else:
                prev_tt = self.voyage_lookup[job_id][current_idx - 1]['tt_next']
                arrival_time = due_date_tracker[job_id] + prev_tt

            berths = port_time_tracker[port_name]
            berth_idx = min(range(len(berths)), key=lambda i: berths[i])
            berth_ready = berths[berth_idx]

            # 2. Antrean Dermaga
            t_dock = max(arrival_time, berth_ready)
            W_cong = max(0, berth_ready - arrival_time)
            actual_start = int(t_dock)

            # 3. Penyesuaian Jendela Pasang Surut (Menggunakan Siklus Modulo)
            tidal_array = self.global_tidal_lookup.get(port_name, {}).get(self.ship_names[job_id], [])
            max_t = len(tidal_array)
            
            if max_t == 0:
                actual_finish = actual_start + proc
            else:
                while True:
                    # Safeguard: Cegah pencarian Infinite Loop jika durasi proses > pasang tertinggi
                    if actual_start - int(t_dock) > max_t * 2:
                        return 1e10 if not return_details else (1e10, [])

                    if buf > 0:
                        start_check = max(0, actual_start - buf)
                        if all(tidal_array[t % max_t] for t in range(start_check, actual_start)):
                            tentative_finish = actual_start + proc
                            actual_departure = tentative_finish
                            while not all(tidal_array[t % max_t] for t in range(actual_departure, actual_departure + buf)):
                                actual_departure += 1
                            actual_finish = actual_departure
                            break
                        else:
                            actual_start += 1
                    else:
                        if all(tidal_array[t % max_t] for t in range(actual_start, actual_start + proc)):
                            actual_finish = actual_start + proc
                            break
                        else:
                            actual_start += 1

            W_tidal = actual_start - int(t_dock)
            C_ij = actual_finish

            # 4. Injeksi Slack (Pembuatan Jadwal Due Date Publik)
            x_slack_val = np.clip(slack_map[(job_id, current_idx)], 0.0, 1.0)
            max_slack = min(24.0, 0.5 * tt_next) if tt_next > 0 else 12.0
            q_ij = x_slack_val * max_slack
            d_ij = C_ij + q_ij

            # Pembaruan Status
            due_date_tracker[job_id] = d_ij
            ship_time_tracker[job_id] = C_ij
            port_time_tracker[port_name][berth_idx] = C_ij
            op_tracker[job_id] += 1

            global_W_cong += W_cong
            global_W_tidal += W_tidal
            global_q += q_ij

            if return_details:
                timetable.append({
                    'job_id': job_id, 'ship_name': self.ship_names[job_id], 'port_name': port_name, 'op_seq': current_idx,
                    'arrival_time': arrival_time, 'actual_start': actual_start, 'C_ij': C_ij, 
                    'slack_injected': q_ij, 'due_date': d_ij, 'W_cong': W_cong, 'W_tidal': W_tidal
                })

        # 5. Evaluasi Kinerja (Job-Level Tardiness)
        total_tardiness = 0.0
        for job_id in self.job_ids:
            Delta_i = ship_time_tracker[job_id] - ship_first_arrival[job_id]
            T_i = max(0, Delta_i - self.D_star[job_id])
            total_tardiness += T_i

        # Cost Function Berbasis Strategi Minmax Penalti
        Z = (alpha * total_tardiness) + (beta * (global_W_cong + global_W_tidal)) + (gamma * global_q)

        if return_details:
            return Z, timetable
        return Z