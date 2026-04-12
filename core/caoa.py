import time
import numpy as np
 
 
def CAOA(N, max_iter, lb, ub, dim, fobj,
         alpha=0.3, beta=0.1, gamma=0.1, delta=1e-3, initial_energy=10.0,
         verbose_interval=50):
 
    # -----------------------------------------------------------------------
    # Persiapan batas pencarian (Eq. 2)
    # -----------------------------------------------------------------------
    lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub, dtype=float)
 
    # -----------------------------------------------------------------------
    # Inisialisasi populasi  -  Eq. 2
    #     x_{i,j} = lb_j + r * (ub_j - lb_j),   r ~ U[0,1]
    # -----------------------------------------------------------------------
    pos = lb + (ub - lb) * np.random.rand(N, dim)   # shape: (N, dim)
 
    # -----------------------------------------------------------------------
    # Inisialisasi energi  -  Eq. 3
    #     E_i = E_init,  ∀ i ∈ {1, …, N}
    # -----------------------------------------------------------------------
    energies = np.full(N, initial_energy, dtype=float)
 
    # -----------------------------------------------------------------------
    # Evaluasi fitness awal & identifikasi global best (gBest)
    # -----------------------------------------------------------------------
    fitness = np.array([fobj(pos[i]) for i in range(N)], dtype=float)
 
    best_idx   = np.argmin(fitness)
    gBestScore = fitness[best_idx]
    gBest      = pos[best_idx].copy()
    cg_curve   = np.zeros(max_iter)
 
    # -----------------------------------------------------------------------
    # Header log
    # -----------------------------------------------------------------------
    print(f"{'Iter':<10} | {'Runtime (s)':<12} | {'Depleted':<10} | "
          f"{'Pop Size':<10} | {'Best Fitness':<20}")
    print("-" * 80)
    start_time = time.time()
 
    # =======================================================================
    # MAIN LOOP  -  Algorithm 1, baris: for t = 1 to T do
    # =======================================================================
    for t in range(max_iter):
 
        # -------------------------------------------------------------------
        # (a) Simpan posisi dan fitness iterasi sebelumnya
        #     "Store previous positions and fitness values"  (Algorithm 1)
        # -------------------------------------------------------------------
        old_positions = pos.copy()       # x_old
        old_fitness   = fitness.copy()   # f_old
 
        # -------------------------------------------------------------------
        # (b) Seleksi Pemimpin berbasis probabilitas inverse-fitness  -  Eq. 6
        #
        #     P(x_i) = 1 / (1 + f(x_i))
        #
        #     Paper menggunakan model probabilistik sehingga leader dipilih
        #     secara stokastik (np.random.choice), bukan selalu argmin.
        #     Normalisasi diperlukan agar total probabilitas = 1.
        #     Catatan: paper mengasumsikan f(x_i) ≥ 0; untuk fungsi dengan
        #     nilai negatif gunakan shift  f(x_i) - min(fitness) + ε agar
        #     penyebut tidak nol atau negatif.
        # -------------------------------------------------------------------
        f_shifted = fitness - np.min(fitness)              # ≥ 0
        probs     = 1.0 / (1.0 + f_shifted)               # Eq. 6
        probs    /= probs.sum()                            # normalisasi
        leader_idx      = np.random.choice(N, p=probs)
        leader_position = pos[leader_idx].copy()           # best_j^t  (Eq. 9)
 
        # -------------------------------------------------------------------
        # (c) Pembaruan posisi setiap buaya  -  Algorithm 1 inner for-loop
        # -------------------------------------------------------------------
        n_depleted_count = 0
 
        for i in range(N):
 
            # "if i is leader then ;" → leader tidak diperbarui (di-skip)
            if i == leader_idx:
                continue
 
            # ---------------------------------------------------------------
            # Mekanisme Berburu (Hunting Mechanism)  -  Eq. 9
            #
            #   x_i^{t+1} = x_i^t
            #               + α · (best_j^t − x_{i,j}^t)   ← eksploitasi
            #               + β · (1 − 2r)                  ← eksplorasi
            #
            #   r ~ U[0,1],  dim-dimensional
            # ---------------------------------------------------------------
            r       = np.random.rand(dim)
            new_pos = (pos[i]
                       + alpha * (leader_position - pos[i])
                       + beta  * (1.0 - 2.0 * r))
 
            # ---------------------------------------------------------------
            # Perbaikan batas (boundary repair)
            #   "Apply boundary control to x_i"  (Algorithm 1)
            # ---------------------------------------------------------------
            new_pos = np.clip(new_pos, lb, ub)
 
            # ---------------------------------------------------------------
            # Evaluasi fitness baru
            #   "Compute new fitness f_new = f(x_i)"  (Algorithm 1)
            # ---------------------------------------------------------------
            f_new = fobj(new_pos)
 
            # ---------------------------------------------------------------
            # Timing of Attacks / Threshold-based reinitialization  -  Eq. 7-8
            #
            #   Δf(x_i) = f_current(x_i) − f_previous(x_i)          (Eq. 7)
            #
            #   Kondisi reinisialisasi (Algorithm 1):
            #     if |f_new − f_old| > δ  AND  f_new > f_old  then
            #         reinitialize x_i randomly within bounds
            #
            #   Artinya: hanya jika fitness MEMBURUK secara signifikan
            #   (melampaui ambang batas δ), buaya dipaksa pindah ke posisi
            #   acak baru untuk menghindari area yang memburuk (Eq. 8).
            # ---------------------------------------------------------------
            delta_f = f_new - old_fitness[i]          # Δf  (Eq. 7)
 
            if abs(delta_f) > delta and f_new > old_fitness[i]:
                # "Reinitialize x_i randomly within bounds; Recompute f_new"
                new_pos = lb + (ub - lb) * np.random.rand(dim)
                new_pos = np.clip(new_pos, lb, ub)    # batas tetap diterapkan
                f_new   = fobj(new_pos)
 
            # ---------------------------------------------------------------
            # "Update x_i and f(x_i)"  (Algorithm 1)
            # ---------------------------------------------------------------
            pos[i]     = new_pos
            fitness[i] = f_new
 
            # ---------------------------------------------------------------
            # Peluruhan Energi  -  Eq. 4 & 5
            #
            #   ‖Δx_i‖ = sqrt( Σ_j (x_{i,j}^{t+1} − x_{i,j}^t)^2 )  (Eq. 5)
            #   E_i^{t+1} = E_i^t − γ · ‖Δx_i‖                       (Eq. 4)
            #
            #   "Compute energy decay: E_i ← E_i − γ · distance(x_i, x_old)"
            # ---------------------------------------------------------------
            dist          = np.sqrt(np.sum((pos[i] - old_positions[i]) ** 2))
            energies[i]  -= gamma * dist
 
            # ---------------------------------------------------------------
            # Pemulihan Energi (Energy Reinitialisation)
            #   "if E_i ≤ 0 then Reinitialize x_i and E_i; Recompute fitness"
            # ---------------------------------------------------------------
            if energies[i] <= 0:
                n_depleted_count += 1
                pos[i]      = lb + (ub - lb) * np.random.rand(dim)
                pos[i]      = np.clip(pos[i], lb, ub)
                energies[i] = initial_energy
                fitness[i]  = fobj(pos[i])
 
        # -------------------------------------------------------------------
        # (d) Perbarui global best  -  Eq. 12
        #     x_global = argmin_{x_i ∈ P} f(x_i)
        #     "Update global best if better solution found"  (Algorithm 1)
        # -------------------------------------------------------------------
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < gBestScore:
            gBestScore = fitness[min_idx]
            gBest      = pos[min_idx].copy()
 
        # -------------------------------------------------------------------
        # (e) Catat kurva konvergensi
        #     "Record gBestScore in convergence curve"  (Algorithm 1)
        # -------------------------------------------------------------------
        cg_curve[t] = gBestScore
 
        # -------------------------------------------------------------------
        # Log konsol
        # -------------------------------------------------------------------
        if (t + 1) % verbose_interval == 0 or t == 0:
            elapsed = time.time() - start_time
            print(f"{t+1:<10} | {elapsed:<12.4f} | {n_depleted_count:<10} | "
                  f"{N:<10} | {gBestScore:.6e}")
 
    # -----------------------------------------------------------------------
    # "return gBest, gBestScore, convergence history"  (Algorithm 1)
    # -----------------------------------------------------------------------
    return gBestScore, gBest, cg_curve