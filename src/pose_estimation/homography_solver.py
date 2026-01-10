# src/pose_estimation/homography_solver.py

def solve_homography(image_points_2d, object_points_3d, camera_matrix):
    """
    Risolve posa usando omografia da 4 punti complanari.
    
    Formula: [r1 r2 t] = K⁻¹ H
    """
    # Calcola omografia
    H, _ = cv2.findHomography(object_points_3d[:, :2], image_points_2d)
    
    # Decomposizione: K⁻¹ H
    K_inv = np.linalg.inv(camera_matrix)
    H_norm = K_inv @ H
    
    # Estrai rotazione e traslazione
    # [r1 r2 t] = H_norm
    r1 = H_norm[:, 0]
    r2 = H_norm[:, 1]
    t = H_norm[:, 2]
    
    # Normalizza
    scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    r1 = r1 / scale
    r2 = r2 / scale
    t = t / scale
    
    # Calcola r3 (perpendicolare)
    r3 = np.cross(r1, r2)
    
    # Costruisci matrice rotazione
    R = np.column_stack([r1, r2, r3])
    
    # Assicura che sia ortogonale (SVD)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Converti a Rodrigues
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    return rvec, tvec