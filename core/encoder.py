import pandas as pd
import numpy as np

class ROVEncoder:
    def __init__(self, voyage_csv_path):
        # Sortir secara absolut untuk konsistensi operasional
        df = pd.read_csv(voyage_csv_path).sort_values(by=['job_id', 'op_seq'])
        op_counts = df.groupby('job_id').size()
        
        base_list = []
        op_map = []
        for job_id, count in op_counts.items():
            base_list.extend([job_id] * count)
            # Simpan indeks (job_id, op_idx) berurutan untuk mapping variabel Slack
            op_map.extend([(job_id, seq) for seq in range(count)])
            
        self.base_job_array = np.array(base_list, dtype=np.int32)
        self.op_map = op_map
        
        # N_w (Jumlah Operasi Aktual)
        self.dim_single = len(self.base_job_array)
        # Dimensi Aktual CAOA (ROV + Slack)
        self.dim = self.dim_single * 2 
        
    def decode(self, continuous_vector):
        """
        Input: Vektor CAOA dimensi 2*N_w (Range: 0.0 s/d 1.0)
        Output: 
          1. Jadwal legal (1D array dari job_id)
          2. Slack dictionary { (job_id, op_idx): nilai_slack_kontinu }
        """
        # Split kromosom metaheuristik
        x_rov = continuous_vector[:self.dim_single]
        x_slack = continuous_vector[self.dim_single:]
        
        # Decode Rank-Order Value (Fase 1)
        sort_indices = np.argsort(x_rov)
        legal_schedule = self.base_job_array[sort_indices]
        
        # Decode Slack Mapping (Fase 2)
        slack_map = {self.op_map[i]: x_slack[i] for i in range(self.dim_single)}
        
        return legal_schedule, slack_map