"""
Vanishing Point Solver - FIX DISTANZA + SMOOTHING
Sistema di stima posa con 6 punti PnP (5 fari + centro outer)

FIX PRINCIPALE: tvec viene ora letto direttamente da solvePnP invece
di essere ricalcolato geometricamente con la formula Z = W*f/d, che era
errata perché:
  1. Confondeva Z (profondità lungo asse ottico) con distanza euclidea lungo il raggio
  2. Assumeva i fari perpendicolari alla linea di vista della camera
  3. Con camera laterale rispetto all'auto, l'ampiezza apparente dei fari
     varia con lo yaw relativo → stima Z sistematicamente sovrastimata → bbox troppo piccola
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Union, List
from collections import deque


class VanishingPointSolver:
    """VP Solver con PnP diretto e smoothing temporale."""

    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict,
                 distortion_coeffs: Optional[np.ndarray] = None):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        self.dist_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5, dtype=np.float32)

        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

        # ===== PARSING YAML =====
        pnp_cfg = vehicle_model.get('vehicle', {}).get('pnp_points_3d', {})
        self.map_3d = {
            'origin':   np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'l_outer':  np.array(pnp_cfg.get('light_l_outer'), dtype=np.float32),
            'r_outer':  np.array(pnp_cfg.get('light_r_outer'), dtype=np.float32),
            'l_top':    np.array(pnp_cfg.get('light_l_top'),   dtype=np.float32),
            'r_top':    np.array(pnp_cfg.get('light_r_top'),   dtype=np.float32),
            'l_bottom': np.array(pnp_cfg.get('light_l_bottom'), dtype=np.float32)
        }

        # Parametri derivati
        self.center_outer_3d = (self.map_3d['l_outer'] + self.map_3d['r_outer']) / 2.0
        self.lights_to_origin_offset = self.map_3d['origin'] - self.center_outer_3d
        self.lights_distance_real = np.linalg.norm(
            self.map_3d['r_outer'] - self.map_3d['l_outer']
        )
        self.lights_height = self.center_outer_3d[2]
        # Fattore di scala correttivo (tunable: 0.80-0.95 se distanza è sovrastimata)
        self.tvec_scale = 1.0

        # Dimensioni veicolo
        vehicle_data = vehicle_model.get('vehicle', {})
        dimensions = vehicle_data.get('dimensions', {})
        self.vehicle_length = dimensions.get('length', 3.70)
        self.vehicle_width = dimensions.get('width', 1.74)
        self.vehicle_height = dimensions.get('height', 1.525)

        # ===== SMOOTHING TEMPORALE =====
        self.tvec_history = deque(maxlen=5)
        self.rvec_history = deque(maxlen=5)
        self.yaw_history = deque(maxlen=20)
        self.yaw_smooth = None
        self.alpha_yaw = 0.25          # EMA smoothing yaw (0=fisso, 1=no smooth)
        self.prev_yaw = 0.0
        
        self.max_reproj_error = 15.0   # pixels — soglia rifiuto posa
        self.last_reproj_error = 0.0   # per debug

        self.alpha_translation = 0.65
        self.alpha_rotation = 0.35

        # ===== PARAMETRI TTI =====
        self.yaw_translation_threshold = np.radians(8)
        self.tti_min = 0.5
        self.tti_max = 30.0
        self.prev_distance = None

        self.vy_history = deque(maxlen=5)
        self.reference_x_axis = None
        self.rotation_counter = 0
        self.translation_counter = 0
        self.rotation_angle_threshold = np.radians(12)
        self.persistence_frames = 3

        self.prev_center_3d = None
        self.prev_tvec_smooth = None
        self.prev_R_smooth = None

        self.debug_mode = True

        print(f"[VP Solver] FIX DISTANZA: usa tvec_pnp diretto da solvePnP")
        print(f"  ✅ Rotation (R): PnP con 6 punti (5 fari + centro outer)")
        print(f"  ✅ Translation: tvec_pnp diretto (NO formula Z=W*f/d)")
        print(f"  ✅ Distanza: |tvec_pnp| (corretta per camera laterale/elevata)")
        print(f"  ✅ Smoothing: EMA + SLERP (alpha_t={self.alpha_translation}, alpha_r={self.alpha_rotation})")
        print(f"  📏 Outer-outer: {self.lights_distance_real:.3f}m")

    def _ensure_numpy_array(self, data: Union[Tuple, List, np.ndarray],
                             shape: Tuple[int, int]) -> np.ndarray:
        if isinstance(data, (tuple, list)):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        else:
            raise TypeError(f"Tipo non supportato: {type(data)}")
        if arr.shape != shape:
            arr = arr.reshape(shape)
        return arr

    def estimate_distance_from_outer_points(self, outer_points: np.ndarray) -> float:
        """
        Stima APPROSSIMATIVA distanza da larghezza apparente fari.

        NOTA: questo metodo è mantenuto solo come fallback/stima iniziale.
        La distanza accurata viene calcolata da |tvec_pnp| in reconstruct_pose_robust.

        LIMITI NOTI:
        - Formula Z = W*f/d assume fari perpendicolari alla linea di vista
        - Con camera laterale, l'ampiezza apparente varia con lo yaw → Z sovrastimato
        - Z è profondità lungo asse ottico, non distanza euclidea → errore aggiuntivo
        """
        L, R = outer_points[0], outer_points[1]
        pixel_dist = np.linalg.norm(R - L)
        if pixel_dist < 1.0:
            return 10.0
        f_avg = (self.fx + self.fy) / 2.0
        Z = (self.lights_distance_real * f_avg) / pixel_dist
        return max(2.0, min(50.0, Z))

    def estimate_yaw_from_lights_geometry(self, outer_points: np.ndarray, distance: float) -> float:
        """Stima yaw da geometria fari."""
        L_2d, R_2d = outer_points[0], outer_points[1]

        L_ray = self.K_inv @ np.append(L_2d, 1.0)
        R_ray = self.K_inv @ np.append(R_2d, 1.0)

        if L_2d[0] > R_2d[0]:
            L_2d, R_2d = R_2d, L_2d

        L_ray = L_ray / np.linalg.norm(L_ray)
        R_ray = R_ray / np.linalg.norm(R_ray)

        L_3d = L_ray * distance
        R_3d = R_ray * distance

        y_dir_cam_3d = R_3d - L_3d
        y_horizontal = np.array([y_dir_cam_3d[0], 0.0, y_dir_cam_3d[2]])

        norm = np.linalg.norm(y_horizontal)
        if norm < 1e-6:
            return self.prev_yaw

        y_horizontal = y_horizontal / norm
        y_veh_horizontal = -y_horizontal
        y_angle = np.arctan2(y_veh_horizontal[0], y_veh_horizontal[2])
        yaw = y_angle + np.pi / 2

        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi

        return yaw

    def estimate_yaw_from_plate_bottom(self, plate_bottom: np.ndarray, distance: float) -> float:
        """Stima yaw da plate bottom."""
        BL, BR = plate_bottom[0], plate_bottom[1]

        BL_ray = self.K_inv @ np.append(BL, 1.0)
        BR_ray = self.K_inv @ np.append(BR, 1.0)

        BL_ray = BL_ray / np.linalg.norm(BL_ray)
        BR_ray = BR_ray / np.linalg.norm(BR_ray)

        BL_3d = BL_ray * distance
        BR_3d = BR_ray * distance

        y_dir_cam_3d = BR_3d - BL_3d
        y_horizontal = np.array([y_dir_cam_3d[0], 0.0, y_dir_cam_3d[2]])

        norm = np.linalg.norm(y_horizontal)
        if norm < 1e-6:
            return self.prev_yaw

        y_horizontal = y_horizontal / norm
        y_veh_horizontal = -y_horizontal
        y_angle = np.arctan2(y_veh_horizontal[0], y_veh_horizontal[2])
        yaw = y_angle + np.pi / 2

        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi

        return yaw

    def slerp_rotation(self, R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
        """Spherical Linear Interpolation tra due matrici di rotazione."""
        q1 = self._rotation_to_quaternion(R1)
        q2 = self._rotation_to_quaternion(R2)

        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-6:
            q_interp = (1 - t) * q1 + t * q2
        else:
            q_interp = (np.sin((1 - t) * theta) / np.sin(theta)) * q1 + \
                       (np.sin(t * theta) / np.sin(theta)) * q2

        q_interp = q_interp / np.linalg.norm(q_interp)
        return self._quaternion_to_rotation(q_interp)

    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Converti matrice 3x3 in quaternione [w, x, y, z]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])

    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """Converti quaternione [w, x, y, z] in matrice 3x3."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def apply_temporal_smoothing(self, tvec_raw: np.ndarray,
                                  R_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applica smoothing temporale: EMA per traslazione, SLERP per rotazione."""
        if self.prev_tvec_smooth is None:
            tvec_smooth = tvec_raw.copy()
        else:
            tvec_smooth = self.alpha_translation * tvec_raw + \
                          (1 - self.alpha_translation) * self.prev_tvec_smooth

        if self.prev_R_smooth is None:
            R_smooth = R_raw.copy()
        else:
            R_smooth = self.slerp_rotation(self.prev_R_smooth, R_raw, self.alpha_rotation)

        self.prev_tvec_smooth = tvec_smooth.copy()
        self.prev_R_smooth = R_smooth.copy()

        return tvec_smooth, R_smooth

    def reconstruct_pose_robust(self, features_2d, frame_idx):
        """
        Stima posa con PnP diretto (6 punti).

        FIX DISTANZA:
        Prima il tvec veniva calcolato con:
          Z = W*f/pixel_dist  →  back-project centro fari  →  + offset geometrico
        Questo era errato per camera laterale/elevata. Ora si usa direttamente
        tvec_pnp da solvePnP, che considera correttamente tutta la geometria.

        Args:
            features_2d: Dict con 'outer', 'top', 'bottom'
            frame_idx: Indice frame

        Returns:
            (rvec_smooth, tvec_smooth, R_smooth) o (None, None, None)
        """
        outer_points = features_2d.get('outer')
        if outer_points is None or len(outer_points) < 2:
            return None, None, None

        # ============================================================
        # PNP CON 6 PUNTI (5 FARI + PUNTO MEDIO OUTER)
        # ============================================================
        list_3d = []
        list_2d = []

        mapping = [
            (features_2d.get('outer', [None, None])[0], 'l_outer'),
            (features_2d.get('outer', [None, None])[1], 'r_outer'),
            (features_2d.get('top', [None, None])[0],   'l_top'),
            (features_2d.get('top', [None, None])[1],   'r_top'),
            (features_2d.get('bottom', [None])[0],      'l_bottom')
        ]

        for pt_2d, key_3d in mapping:
            if pt_2d is not None:
                list_2d.append(pt_2d)
                list_3d.append(self.map_3d[key_3d])

        # 6° punto: centro outer
        if len(outer_points) == 2:
            center_outer_2d = np.mean(outer_points, axis=0)
            list_2d.append(center_outer_2d)
            center_outer_3d = (self.map_3d['l_outer'] + self.map_3d['r_outer']) / 2.0
            list_3d.append(center_outer_3d)

        if len(list_2d) < 4:
            return None, None, None

        img_pts = np.array(list_2d, dtype=np.float32)
        obj_pts = np.array(list_3d, dtype=np.float32)

        success, rvec_pnp, tvec_pnp = cv2.solvePnP(
            obj_pts, img_pts, self.K, self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
   
        if not success:
            return None, None, None
        
        # ============================================================
        # REPROJECTION ERROR CHECK
        # Se l'errore è alto, i 2D points sono spazzatura (tracker drift)
        # → rifiuta questa posa, il chiamante userà last_good_pose
        # ============================================================
        projected, _ = cv2.projectPoints(
            obj_pts, rvec_pnp, tvec_pnp, self.K, self.dist_coeffs
        )
        reproj_errors = np.linalg.norm(
            img_pts - projected.reshape(-1, 2), axis=1
        )
        mean_reproj_error = float(np.mean(reproj_errors))
        self.last_reproj_error = mean_reproj_error   # esposto per debug

        if mean_reproj_error > self.max_reproj_error:
            if self.debug_mode:
                print(f"  ❌ Frame {frame_idx}: reproj_error={mean_reproj_error:.1f}px "
                    f"> {self.max_reproj_error}px → pose rejected")
            return None, None, None

        R_pnp, _ = cv2.Rodrigues(rvec_pnp)

        # ============================================================
        # FIX: USA tvec_pnp DIRETTAMENTE
        #
        # solvePnP calcola la trasformazione che mappa i punti dall'object
        # frame (veicolo) al camera frame: P_cam = R @ P_obj + t
        # Quindi tvec_pnp = posizione dell'origine del veicolo (0,0,0)
        # nel camera frame → distanza corretta in qualsiasi geometria.
        #
        # L'approccio geometrico precedente era errato perché:
        #   1. Z = W*f/d assume i fari perpendicolari alla vista (falso con camera laterale)
        #   2. Usava Z come distanza euclidea lungo il raggio (sbagliato: d_eucl = Z/cos θ)
        #   3. Distanza sistematicamente sovrastimata → bbox troppo piccola
        # ============================================================
        tvec_final = tvec_pnp.reshape(3, 1).astype(np.float32) * self.tvec_scale

        # ============================================================
        # SMOOTHING TEMPORALE
        # ============================================================
        tvec_smooth, R_smooth = self.apply_temporal_smoothing(
            tvec_final.flatten(), R_pnp
        )

        rvec_smooth, _ = cv2.Rodrigues(R_smooth)
        tvec_smooth = tvec_smooth.reshape(3, 1).astype(np.float32)

        if self.debug_mode and frame_idx % 10 == 0:
            pitch_from_R = np.arctan2(-R_pnp[2, 1], R_pnp[2, 2])
            distance_pnp = float(np.linalg.norm(tvec_pnp))
            # Distanza orizzontale (elimina componente verticale camera-veicolo)
            dist_horizontal = float(np.sqrt(float(tvec_pnp[0])**2 + float(tvec_pnp[2])**2))
            print(f"Frame {frame_idx}: pitch_pnp={np.degrees(pitch_from_R):.1f}°, "
                f"dist_pnp={distance_pnp:.2f}m, dist_horiz={dist_horizontal:.2f}m, npts={len(list_2d)}")

        return rvec_smooth, tvec_smooth, R_smooth

    def extract_yaw_from_rotation(self, R: np.ndarray) -> float:
        """Estrae yaw da matrice rotazione."""
        x_axis = R[:, 0].copy()
        x_axis[1] = 0.0
        norm = np.linalg.norm(x_axis)
        if norm < 1e-6:
            return 0.0
        x_axis /= norm
        return np.arctan2(x_axis[0], x_axis[2])

    def classify_motion_type(self, R: np.ndarray) -> str:
        """Classifica TRANSLATION vs STEERING su finestra 20 frame di yaw."""
        raw_yaw = self.extract_yaw_from_rotation(R)

        # EMA smoothing dello yaw
        if self.yaw_smooth is None:
            self.yaw_smooth = raw_yaw
        else:
            # Gestisci wrap-around ±π
            diff = raw_yaw - self.yaw_smooth
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            self.yaw_smooth = self.yaw_smooth + self.alpha_yaw * diff

        self.yaw_history.append(self.yaw_smooth)

        if len(self.yaw_history) < 3:
            return "TRANSLATION"

        yaw_range = max(self.yaw_history) - min(self.yaw_history)
        if yaw_range > np.radians(5):
            return "STEERING"
        return "TRANSLATION"

    def calculate_tti(self, distance: float, dt: float) -> Optional[float]:
        """Calcola TTI (Time To Impact)."""
        if self.prev_distance is None:
            self.prev_distance = distance
            return None

        V = (distance - self.prev_distance) / dt

        if abs(V) > 5.0:
            self.prev_distance = distance
            return None
        if V >= -0.01:
            self.prev_distance = distance
            return None

        tti = -distance / V
        self.prev_distance = distance
        return tti

    def validate_pose_with_tti(self, tti: Optional[float]) -> Tuple[bool, str]:
        """Valida posa usando TTI."""
        if tti is None:
            return True, "TTI not available"
        if tti < 0:
            if abs(tti) > self.tti_max:
                return False, f"Moving away too slowly (TTI={tti:.1f}s)"
            return True, f"Moving away (TTI={tti:.1f}s)"
        if tti < self.tti_min:
            return False, f"TTI too small (TTI={tti:.2f}s)"
        if tti > self.tti_max:
            return False, f"TTI too large (TTI={tti:.1f}s)"
        return True, f"TTI valid ({tti:.1f}s)"

    def reset_tti_history(self):
        self.prev_distance = None

    def reset_temporal_smoothing(self):
        self.vy_history.clear()
        self.yaw_history.clear()
        self.tvec_history.clear()
        self.rvec_history.clear()
        self.prev_tvec_smooth = None
        self.prev_R_smooth = None

    def reset_vp_persistence(self):
        self.prev_center_3d = None
        self.prev_yaw = 0.0
        self.reference_x_axis = None
        self.rotation_counter = 0
        self.translation_counter = 0
        self.reset_temporal_smoothing()
        print("[VP Solver] Reset")

    def estimate_pose_multifeature(
        self,
        features_t2: Dict[str, np.ndarray],
        plate_bottom_t2: Optional[np.ndarray],
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """
        Stima posa con PnP diretto e smoothing.

        FIX: la distanza viene ora calcolata come |tvec_pnp|, non dalla
        formula approssimata W*f/d che era inaccurata con camera laterale.
        """
        outer_points = features_t2.get('outer')
        if outer_points is None:
            return None

        if plate_bottom_t2 is not None:
            features_t2['plate_bottom'] = plate_bottom_t2

        # Stima posa
        res = self.reconstruct_pose_robust(features_t2, frame_idx)
        if res[0] is None:
            return None

        rvec, tvec, R = res

        # ============================================================
        # FIX: distanza da |tvec| (corretta), non dalla formula
        # ============================================================
        distance = float(np.linalg.norm(tvec))

        motion_type = self.classify_motion_type(R)

        tti = self.calculate_tti(distance, dt=1.0 / 30.0)
        tti_valid, tti_msg = self.validate_pose_with_tti(tti)

        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'distance': distance,
            'frame': frame_idx,
            'is_valid': True,
            'motion_type': motion_type,
            'tti': tti,
            'tti_valid': tti_valid,
            'debug': {
                'method': 'pnp6_direct_tvec',
                'yaw_degrees': np.degrees(self.yaw_smooth) if self.yaw_smooth is not None else np.degrees(self.extract_yaw_from_rotation(R)),
                'pitch_degrees': np.degrees(np.arctan2(-R[2, 1], R[2, 2])),
                'has_plate': plate_bottom_t2 is not None,
                'smoothing': f'EMA(t={self.alpha_translation})+SLERP(r={self.alpha_rotation})'
            }
        }

    def estimate_pose(
        self,
        lights_frame: Union[Tuple, np.ndarray],
        plate_bottom: Optional[np.ndarray],
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """Backward compatibility."""
        features = {'outer': self._ensure_numpy_array(lights_frame, (2, 2))}
        return self.estimate_pose_multifeature(features, plate_bottom, frame_idx)