import pandas as pd
import numpy as np

class ROVEncoder:
    def __init__(self, voyage_csv_path):
        # Inisialisasi base job array
        self.base_job_array = self._build_base_array(voyage_csv_path)
        
        # Dimensi dihitung dari banyaknya operasi
        self.dim = len(self.base_job_array)
        
    def _build_base_array(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # Hitung jumlah operasi (op_seq) per kapal (job_id)
        # Mengembalikan series dengan index = job_id, value = jumlah operasi
        op_counts = df.groupby('job_id').size()
        
        # Bangun Base Array
        # Contoh: Jika job 1 punya 3 operasi, dan job 2 punya 2 operasi
        # Hasilnya: [1, 1, 1, 2, 2]
        base_list = []
        for job_id, count in op_counts.items():
            base_list.extend([job_id] * count)
            
        return np.array(base_list, dtype=np.int32)

    def decode(self, continuous_vector):
        """
        Input: Vektor kontinu dari CAOA (ukuran = self.dim)
        Output: Permutasi jadwal JSSP diskrit yang 100% legal.
        """
        sort_indices = np.argsort(continuous_vector)
        
        # Re-order base array
        discrete_permutation = self.base_job_array[sort_indices]
        
        return discrete_permutation

# --- Testing ---
if __name__ == "__main__":
    # Inisialisasi Encoder
    encoder = ROVEncoder('data/processed/voyage_p3.csv')
    
    print(f"Dimensi Problem (Total Operasi): {encoder.dim}")
    
    # Simulasi: CAOA menghasilkan satu vektor posisi acak
    dummy_caoa_vector = np.random.uniform(-1.0, 1.0, encoder.dim)
    
    # Translasi vektor ke urutan jadwal
    legal_schedule = encoder.decode(dummy_caoa_vector)
    
    print(f"Sample Permutasi Jadwal (100 pertama):\n{legal_schedule[:100]}")