# src/pose_estimation/translational_solver.py

def solve_translational_motion(lights_frame1, lights_frame2, camera_matrix):
    """
    Risolve posa da movimento traslatorio tra 2 frame.
    
    Args:
        lights_frame1: ((L1_x, L1_y), (R1_x, R1_y)) frame 1
        lights_frame2: ((L2_x, L2_y), (R2_x, R2_y)) frame 2
        camera_matrix: K
    """
    L1, R1 = lights_frame1
    L2, R2 = lights_frame2
    
    # 1. Trova punto di fuga Vx (intersezione traiettorie)
    # Linea L1-L2
    line1 = np.cross([L1[0], L1[1], 1], [L2[0], L2[1], 1])
    # Linea R1-R2
    line2 = np.cross([R1[0], R1[1], 1], [R2[0], R2[1], 1])
    # Intersezione
    Vx = np.cross(line1, line2)
    Vx = Vx[:2] / Vx[2]  # Normalizza coordinate omogenee
    
    # 2. Direzione 3D del movimento
    K_inv = np.linalg.inv(camera_matrix)
    direction_3d = K_inv @ np.array([Vx[0], Vx[1], 1])
    direction_3d = direction_3d / np.linalg.norm(direction_3d)
    
    # 3. Verifica perpendicolarità (traslazione rettilinea)
    # Direzione segmento luci
    segment_dir_image = np.array([R1[0] - L1[0], R1[1] - L1[1], 0])
    segment_dir_3d = K_inv @ np.array([segment_dir_image[0], segment_dir_image[1], 1])
    
    # Prodotto scalare (dovrebbe essere ~0 per traslazione pura)
    dot_product = np.dot(direction_3d, segment_dir_3d)
    
    if abs(dot_product) > 0.1:
        print(f"⚠️ Non è traslazione pura: dot={dot_product:.3f}")
        return None, None
    
    # 4. Calcola distanza piano π
    # [Qui implementi il calcolo geometrico dalla slide 21]
    
    return rvec, tvec