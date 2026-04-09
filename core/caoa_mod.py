import time
import numpy as np

def CAOA(N, max_iter, lb, ub, dim, fobj, 
         alpha=0.146335, beta=0.098863, gamma=0.014926, delta=8, initial_energy=100.0,
         verbose_interval=100, seed_position=None):
    
    if np.isscalar(lb): lb = np.full(dim, lb)
    else: lb = np.array(lb)
    if np.isscalar(ub): ub = np.full(dim, ub)
    else: ub = np.array(ub)
    
    pos = lb + (ub - lb) * np.random.rand(N, dim)

    if seed_position is not None:
        pos[0, :] = seed_position
        print("--> SEED POSITION INJECTED")

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
            new_pos = np.clip(new_pos, lb, ub)

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
                    # Ambush gagal total. Eksplorasi acak, tapi HANYA TERIMA jika lebih baik dari posisi acaknya.
                    rand_pos = lb + (ub - lb) * np.random.rand(dim)
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
            random_positions = lb + (ub - lb) * np.random.rand(n_depleted_count, dim)
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

    return gBestScore, gBest, cg_curve