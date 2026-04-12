import time
import numpy as np

def CAOASSR(N, max_iter, lb, ub, dim, fobj, 
         alpha=0.3, beta=0.1, gamma=0.1, delta=1e-3, initial_energy=10.0,
         verbose_interval=50):
    
    if np.isscalar(lb): lb = np.full(dim, lb)
    else: lb = np.array(lb)
    if np.isscalar(ub): ub = np.full(dim, ub)
    else: ub = np.array(ub)
    
    pos = lb + (ub - lb) * np.random.rand(N, dim)

    energies = initial_energy * np.ones(N)
    fitness = np.zeros(N)
    
    # Evaluasi posisi awal
    for i in range(N):
        fitness[i] = fobj(pos[i, :])
        
    best_idx = np.argmin(fitness)
    gBestScore = fitness[best_idx]
    gBest = pos[best_idx, :].copy()
    cg_curve = np.zeros(max_iter)
    
    print(f"{'Iter':<6} | {'Runtime (s)':<12} | {'Depleted':<9} | {'Best Fitness':<20}")
    print("-" * 60)

    # == Mekanisme SSR ==
    lb_dynamic = np.copy(lb)
    ub_dynamic = np.copy(ub)
    
    # Parameter SSR (Berdasarkan Bab 3.2.3)
    IT = max(1, int(max_iter * 0.05)) # Cek stagnasi
    DTh = 1e-8 # Threshold perbaikan
    delta_min = 0.05 * (ub - lb) # Safety clamp (5% dari rentang inisialisasi)
    
    best_fitness_history = gBestScore
    ssr_active = False # Flag status ruang pencarian
    
    start_time = time.time()

    for t in range(max_iter):
        old_positions = pos.copy()
        n_depleted_count = 0
        
        # 1. KOREKSI PROBABILITAS: Scaling terhadap fitness terbaik agar seleksi rasional
        min_fit_current = np.min(fitness)
        # Kurangi semua dengan min_fit agar yang terbaik memiliki probabilitas sangat dominan
        scaled_fitness = fitness - min_fit_current
        probs = 1.0 / (1.0 + scaled_fitness)
        probs_normalized = probs / np.sum(probs)
        leader_idx = np.random.choice(N, p=probs_normalized) 
        leader_position = pos[leader_idx, :].copy()
        
        # 2. PEMBARUAN POSISI
        for i in range(N):
            if i == leader_idx: continue
            
            r = np.random.rand(dim)
            step_direction = alpha * (leader_position - pos[i, :]) + beta * (1.0 - 2.0 * r)
            new_pos = pos[i, :] + step_direction

            out_upper = new_pos > ub_dynamic
            out_lower = new_pos < lb_dynamic
            
            # Pantulkan sedikit ke dalam batas agar nilai desimal tetap unik untuk diurutkan (argsort)
            noise_scale = 0.001 * (ub_dynamic - lb_dynamic)

            if np.any(out_upper):
                new_pos[out_upper] = ub_dynamic[out_upper] - np.random.rand(np.sum(out_upper)) * noise_scale[out_upper]
            if np.any(out_lower):
                new_pos[out_lower] = lb_dynamic[out_lower] + np.random.rand(np.sum(out_lower)) * noise_scale[out_lower]

            new_fit = fobj(new_pos)
            delta_fit = new_fit - fitness[i]
            
            # KOREKSI AMBUSH & GREEDY ACCEPTANCE
            if delta_fit < 0:
                # Solusi membaik -> Terima tanpa syarat
                pos[i, :] = new_pos
                fitness[i] = new_fit
            else:
                # Solusi memburuk. Cek apakah melebih ambang batas kegagalan ambush (delta)
                if delta_fit > delta:
                    # Ambush gagal total. Regenerasi posisi acak sesuai status SSR.
                    if not ssr_active:
                        rand_pos = lb + (ub - lb) * np.random.rand(dim)
                    else:
                        std_dev = (ub_dynamic - lb_dynamic) / 3.0
                        rand_pos = np.random.normal(loc=gBest, scale=std_dev)
                        rand_pos = np.clip(rand_pos, lb_dynamic, ub_dynamic)
                    
                    rand_fit = fobj(rand_pos)
                    
                    if rand_fit < fitness[i]:
                        pos[i, :] = rand_pos
                        fitness[i] = rand_fit
                        # Putuskan energi agen ini agar langsung di-recharge di fase berikutnya
                        energies[i] = 0
                # Jika 0 <= delta_fit <= delta, buaya diam (tidak menerima posisi buruk, menghemat energi)

        # 3. KOREKSI PELURUHAN ENERGI (Hanya hitung jarak perpindahan aktual, bukan teleportasi acak)
        distances = np.sqrt(np.sum((pos - old_positions)**2, axis=1))
        energies = energies - gamma * distances
        
        # 4. RECOVERY ENERGI
        depleted = energies <= 0
        if np.any(depleted):
            n_depleted_count = np.sum(depleted)

            if not ssr_active:
                random_positions = lb + (ub - lb) * np.random.rand(n_depleted_count, dim)
            else:
                std_dev = (ub_dynamic - lb_dynamic) / 3.0 
                random_positions = np.random.normal(loc=gBest, scale=std_dev, size=(n_depleted_count, dim))
                
                # Pastikan hasil Gaussian tetap berada dalam batas dinamik
                random_positions = np.clip(random_positions, lb_dynamic, ub_dynamic)

            pos[depleted, :] = random_positions
            energies[depleted] = initial_energy
            
            for idx in np.where(depleted)[0]:
                fitness[idx] = fobj(pos[idx, :])

        # 5. UPDATE GLOBAL BEST
        min_fit = np.min(fitness)
        if min_fit < gBestScore:
            gBestScore = min_fit
            gBest = pos[np.argmin(fitness), :].copy()
            
        cg_curve[t] = gBestScore
        
        if verbose_interval > 0 and ((t + 1) % verbose_interval == 0 or t == 0):
            elapsed = time.time() - start_time
            print(f"{t+1:<6} | {elapsed:<12.2f} | {n_depleted_count:<9} | {gBestScore:.6e}")
        
        # Evaluasi Stagnasi & Eksekusi SSR
        if (t + 1) % IT == 0:
            FI = abs(best_fitness_history - gBestScore)
            
            if FI < (0.01 * best_fitness_history):
                ssr_active = True
                print(f"--> [Iter {t+1}] Stagnasi terdeteksi (FI: {FI:.2e}). SSR Diaktifkan/Diperbarui.")
                
                for j in range(dim):
                    range_j = ub_dynamic[j] - lb_dynamic[j]
                    
                    # Cegah pembagian dengan nol
                    if range_j < 1e-10: continue 
                    
                    # Rasio jarak posisi terbaik (Dis_j)
                    dis_j = (ub_dynamic[j] - gBest[j]) / range_j
                    
                    # Interval Reduksi (RI_j) dengan Safety Clamp
                    ri_j = max(range_j * max(dis_j, 1 - dis_j), delta_min[j])
                    
                    # Pembaruan Batas
                    low_baru = gBest[j] - (ri_j / 2)
                    up_baru = gBest[j] + (ri_j / 2)
                    
                    # Boundary Check agar tidak melebihi domain asli
                    lb_dynamic[j] = max(low_baru, lb[j])
                    ub_dynamic[j] = min(up_baru, ub[j])
            
            # Reset history untuk interval pengawasan berikutnya
            best_fitness_history = gBestScore

    return gBestScore, gBest, cg_curve