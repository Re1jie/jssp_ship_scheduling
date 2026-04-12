"""
Ekstensi yang tersedia:
    greedy_accept    : Saat reinisialisasi karena ambush gagal, posisi acak baru
                       hanya diterima jika fitness-nya lebih baik dari posisi lama.
                       Paper: selalu ganti tanpa syarat.

    vectorized_energy: Hitung peluruhan energi semua individu secara vektorisasi
                       SETELAH loop individu selesai (lebih cepat secara komputasi).
                       Paper: hitung di dalam loop, langsung setelah tiap individu.

    idle_on_small_df : Jika |Δf| ≤ δ (perubahan kecil), individu diam — posisi
                       tidak diperbarui, energi tidak berkurang.
                       Paper: tidak ada mekanisme diam; selalu update.
"""

import time
import numpy as np


def CAOA(
    N, max_iter, lb, ub, dim, fobj,
    alpha=0.3, beta=0.1, gamma=0.1, delta=1e-3, initial_energy=10.0,
    # --- Ekstensi ---
    greedy_accept=False,
    vectorized_energy=False,
    idle_on_small_df=False,
    # --- Utilitas ---
    verbose_interval=50,
):
    """
    Parameters
    ----------
    N, max_iter, lb, ub, dim, fobj : standar CAOA (lihat paper Eq. 2-12)
    alpha   : step size eksploitasi          (default paper: 0.3)
    beta    : bobot gangguan acak eksplorasi (default paper: 0.1)
    gamma   : laju peluruhan energi          (default paper: 0.1)
    delta   : ambang batas stabilitas fitness(default paper: 1e-3)
    initial_energy : E_init                  (default paper: 10.0)

    greedy_accept    : bool — ekstensi #1 (default False)
    vectorized_energy: bool — ekstensi #2 (default False)
    idle_on_small_df : bool — ekstensi #3 (default False)

    Returns
    -------
    gBestScore : float
    gBest      : ndarray
    cg_curve   : ndarray
    """

    # ------------------------------------------------------------------
    # Persiapan batas — Eq. 2
    # ------------------------------------------------------------------
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.array(ub, dtype=float)

    # ------------------------------------------------------------------
    # Inisialisasi populasi — Eq. 2
    # ------------------------------------------------------------------
    pos = lb + (ub - lb) * np.random.rand(N, dim)

    # ------------------------------------------------------------------
    # Inisialisasi energi — Eq. 3
    # ------------------------------------------------------------------
    energies = np.full(N, initial_energy, dtype=float)

    # ------------------------------------------------------------------
    # Evaluasi fitness awal & global best — Eq. 12
    # ------------------------------------------------------------------
    fitness = np.array([fobj(pos[i]) for i in range(N)], dtype=float)
    best_idx   = np.argmin(fitness)
    gBestScore = fitness[best_idx]
    gBest      = pos[best_idx].copy()
    cg_curve   = np.zeros(max_iter)

    mode_str = []
    if greedy_accept:     mode_str.append("greedy_accept")
    if vectorized_energy: mode_str.append("vectorized_energy")
    if idle_on_small_df:  mode_str.append("idle_on_small_df")
    mode_label = ", ".join(mode_str) if mode_str else "paper-faithful"

    print(f"Mode: {mode_label}")
    print(f"{'Iter':<6} | {'Runtime (s)':<12} | {'Depleted':<9} | {'Best Fitness':<20}")
    print("-" * 60)
    start_time = time.time()

    # ==================================================================
    # MAIN LOOP — Algorithm 1: for t = 1 to T do
    # ==================================================================
    for t in range(max_iter):

        # --------------------------------------------------------------
        # (a) Simpan posisi dan fitness sebelumnya
        #     "Store previous positions and fitness values"
        # --------------------------------------------------------------
        old_positions = pos.copy()
        old_fitness   = fitness.copy()
        n_depleted    = 0

        # --------------------------------------------------------------
        # (b) Seleksi leader — Eq. 6
        #     P(x_i) = 1 / (1 + f(x_i))
        #     Shift numerik agar penyebut selalu > 0.
        # --------------------------------------------------------------
        f_shifted = fitness - np.min(fitness)
        probs     = 1.0 / (1.0 + f_shifted)
        probs    /= probs.sum()
        leader_idx      = np.random.choice(N, p=probs)
        leader_pos      = pos[leader_idx].copy()

        # --------------------------------------------------------------
        # (c) Loop per-individu — Algorithm 1 inner for-loop
        # --------------------------------------------------------------
        for i in range(N):

            # Leader di-skip — "if i is leader then ;"
            if i == leader_idx:
                continue

            # ----------------------------------------------------------
            # Update posisi — Eq. 9
            #   x_i^{t+1} = x_i^t
            #               + α·(best_j^t − x_{i,j}^t)
            #               + β·(1 − 2r)
            # ----------------------------------------------------------
            r       = np.random.rand(dim)
            new_pos = pos[i] + alpha * (leader_pos - pos[i]) + beta * (1.0 - 2.0 * r)

            # Boundary repair — "Apply boundary control to x_i"
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluasi — "Compute new fitness f_new = f(x_i)"
            f_new   = fobj(new_pos)
            delta_f = f_new - old_fitness[i]          # Δf — Eq. 7

            # ----------------------------------------------------------
            # EKSTENSI #3: idle_on_small_df
            #   Jika |Δf| ≤ δ, individu diam — hemat energi lokal.
            #   TIDAK ada di paper. Skip update posisi dan energi.
            # ----------------------------------------------------------
            if idle_on_small_df and abs(delta_f) <= delta:
                continue                               # pos[i] tetap, energi tidak turun

            # ----------------------------------------------------------
            # Threshold-based reinit — Eq. 8, Algorithm 1
            #   if |Δf| > δ AND f_new > f_old:
            #       reinitialize x_i; recompute f_new
            # ----------------------------------------------------------
            if abs(delta_f) > delta and f_new > old_fitness[i]:

                rand_pos = lb + (ub - lb) * np.random.rand(dim)
                rand_pos = np.clip(rand_pos, lb, ub)
                rand_fit = fobj(rand_pos)

                # ------------------------------------------------------
                # EKSTENSI #1: greedy_accept
                #   Paper: selalu terima posisi acak baru tanpa syarat.
                #   Ekstensi: hanya terima jika rand_fit < fitness[i].
                # ------------------------------------------------------
                if greedy_accept:
                    if rand_fit < fitness[i]:
                        new_pos = rand_pos
                        f_new   = rand_fit
                    # Jika tidak lebih baik: biarkan new_pos hasil Eq. 9
                else:
                    # Perilaku paper — selalu ganti
                    new_pos = rand_pos
                    f_new   = rand_fit

            # "Update x_i and f(x_i)"
            pos[i]     = new_pos
            fitness[i] = f_new

            # ----------------------------------------------------------
            # Peluruhan energi — Eq. 4–5  [PAPER: di dalam loop]
            #   Hanya dijalankan jika vectorized_energy=False (default).
            # ----------------------------------------------------------
            if not vectorized_energy:
                dist         = np.sqrt(np.sum((pos[i] - old_positions[i]) ** 2))
                energies[i] -= gamma * dist

                # Reinit energi — "if E_i ≤ 0 then reinitialize"
                if energies[i] <= 0:
                    n_depleted    += 1
                    pos[i]         = lb + (ub - lb) * np.random.rand(dim)
                    pos[i]         = np.clip(pos[i], lb, ub)
                    energies[i]    = initial_energy
                    fitness[i]     = fobj(pos[i])

        # --------------------------------------------------------------
        # EKSTENSI #2: vectorized_energy
        #   Paper: peluruhan dihitung di dalam loop (blok di atas).
        #   Ekstensi: hitung setelah loop selesai — lebih cepat (NumPy).
        # --------------------------------------------------------------
        if vectorized_energy:
            distances  = np.sqrt(np.sum((pos - old_positions) ** 2, axis=1))
            energies  -= gamma * distances

            depleted = energies <= 0
            if np.any(depleted):
                n_dep    = int(np.sum(depleted))
                n_depleted += n_dep
                rnd_pos  = lb + (ub - lb) * np.random.rand(n_dep, dim)
                pos[depleted]     = np.clip(rnd_pos, lb, ub)
                energies[depleted] = initial_energy
                for idx in np.where(depleted)[0]:
                    fitness[idx] = fobj(pos[idx])

        # --------------------------------------------------------------
        # Update global best — Eq. 12
        #   "Update global best if better solution found"
        # --------------------------------------------------------------
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < gBestScore:
            gBestScore = fitness[min_idx]
            gBest      = pos[min_idx].copy()

        cg_curve[t] = gBestScore

        if verbose_interval > 0 and ((t + 1) % verbose_interval == 0 or t == 0):
            elapsed = time.time() - start_time
            print(f"{t+1:<6} | {elapsed:<12.4f} | {n_depleted:<9} | {gBestScore:.6e}")

    return gBestScore, gBest, cg_curve