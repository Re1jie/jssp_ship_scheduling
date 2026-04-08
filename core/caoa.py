import time
import numpy as np

def CAOA(N, max_iter, lb, ub, dim, fobj, 
         alpha=0.3, beta=0.1, gamma=0.1, delta=1e-3, initial_energy=10.0,
         verbose_interval=10, seed_position=None):
    """
    Menjalankan algoritma CAOA untuk meminimalkan fungsi objektif.
    """
    # Inisialisasi
    if np.isscalar(lb): lb = np.full(dim, lb)
    else: lb = np.array(lb)
    if np.isscalar(ub): ub = np.full(dim, ub)
    else: ub = np.array(ub)
    
    pos = lb + (ub - lb) * np.random.rand(N, dim)

    if seed_position is not None:
        pos[0, :] = seed_position
        print("SEEDED")

    energies = initial_energy * np.ones(N)
    fitness = np.zeros(N)
    
    # Evaluasi posisi awal
    for i in range(N):
        fitness[i] = fobj(pos[i, :])
        
    best_idx = np.argmin(fitness)
    gBestScore = fitness[best_idx]
    gBest = pos[best_idx, :].copy()
    cg_curve = np.zeros(max_iter)
    
    print(f"{'Iter':<10} | {'Runtime (s)':<12} | {'Depleted':<10} | {'Pop Size':<10} | {'Best Fitness':<20}")
    print("-" * 80)
    
    start_time = time.time()

    # Main loop
    for t in range(max_iter):
        old_positions = pos.copy()
        old_fitness = fitness.copy()
        n_depleted_count = 0
        
        # Seleksi Pemimpin Stokastik (Berdasarkan probabilitas dari nilai fitness)
        probs = 1.0 / (1.0 + np.abs(fitness))
        probs_normalized = probs / np.sum(probs)
        leader_idx = np.random.choice(N, p=probs_normalized) 
        leader_position = pos[leader_idx, :].copy()
        
        # Pembaruan Posisi (Pergerakan Buaya)
        for i in range(N):
            if i == leader_idx: continue
            
            r = np.random.rand(dim)
            # Bergerak ke arah pemimpin (alpha) dengan gangguan acak (beta)
            new_pos = pos[i, :] + alpha * (leader_position - pos[i, :]) + beta * (1.0 - 2.0 * r)
            new_pos = np.clip(new_pos, lb, ub)

            # C. Percobaan Penyergapan & Koreksi
            new_fit = fobj(new_pos)
            delta_fit = new_fit - old_fitness[i]
            
            # Jika langkah baru membuat fitness jauh lebih buruk dari batas toleransi (delta),
            # paksa buaya untuk bereksplorasi secara acak di tempat baru (Re-inisialisasi)
            if delta_fit > 0 and abs(delta_fit) > delta:
                new_pos = lb + (ub - lb) * np.random.rand(dim)
                new_fit = fobj(new_pos)
            
            pos[i, :] = new_pos
            fitness[i] = new_fit

        # Peluruhan Energi Adaptif (Semakin jauh bergerak, semakin cepat energi habis)
        distances = np.sqrt(np.sum((pos - old_positions)**2, axis=1))
        energies = energies - gamma * distances
        
        # Mekanisme Pemulihan Energi (Buaya kelelahan beristirahat di posisi acak baru)
        depleted = energies <= 0
        if np.any(depleted):
            n_depleted_count = np.sum(depleted)
            random_positions = lb + (ub - lb) * np.random.rand(n_depleted_count, dim)
            pos[depleted, :] = random_positions
            energies[depleted] = initial_energy
            # Evaluasi ulang fitness untuk agen yang baru di-reset
            for idx in np.where(depleted)[0]:
                fitness[idx] = fobj(pos[idx, :])

        # Pembaruan Global Best
        min_fit = np.min(fitness)
        min_idx = np.argmin(fitness)
        if min_fit < gBestScore:
            gBestScore = min_fit
            gBest = pos[min_idx, :].copy()
            
        cg_curve[t] = gBestScore
        
        # Log konsol
        if (t + 1) % verbose_interval == 0 or t == 0:
            elapsed = time.time() - start_time
            print(f"{t+1:<10} | {elapsed:<12.2f} | {n_depleted_count:<10} | {N:<10} | {gBestScore:.6e}")

    return gBestScore, gBest, cg_curve