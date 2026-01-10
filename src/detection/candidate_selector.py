"""
Candidate Selector - Selezione della coppia di fari posteriori migliore.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from .light_detector import LightCandidate


class CandidateSelector:
    """
    Seleziona la coppia di fari posteriori più probabile tra i candidati rilevati.
    """
    
    def __init__(self, config: Dict, frame_width: int):
        """
        Inizializza il selettore.
        
        Args:
            config: Dizionario di configurazione (da detection_params.yaml)
            frame_width: Larghezza del frame in pixel
        """
        self.frame_width = frame_width
        
        # Parametri di selezione
        sel_cfg = config.get('tail_lights_selection', {})
        self.min_horizontal_distance = sel_cfg.get('min_horizontal_distance', 50)
        self.max_horizontal_distance_ratio = sel_cfg.get('max_horizontal_distance_ratio', 0.8)
        self.max_vertical_offset = sel_cfg.get('max_vertical_offset', 50)
        self.min_area_similarity = sel_cfg.get('min_area_similarity', 0.5)
        self.min_pair_score = sel_cfg.get('min_pair_score', 0.3)
    
    def compute_pair_score(self, c1: LightCandidate, c2: LightCandidate) -> Tuple[float, Dict]:
        """
        Calcola uno score di compatibilità per una coppia di candidati.
        
        Args:
            c1: Primo candidato
            c2: Secondo candidato
            
        Returns:
            Tuple (score, metrics_dict) dove score è tra 0 e 1
        """
        # Distanze
        dx = abs(c1.center[0] - c2.center[0])
        dy = abs(c1.center[1] - c2.center[1])
        
        # Rapporto aree
        area_ratio = min(c1.area, c2.area) / max(c1.area, c2.area)
        
        # Score componenti (tutti tra 0 e 1)
        
        # 1. Distanza orizzontale (vuole essere significativa ma non eccessiva)
        horizontal_score = 0.0
        if dx > self.min_horizontal_distance:
            max_allowed = self.frame_width * self.max_horizontal_distance_ratio
            if dx <= max_allowed:
                # Normalizza tra 0 e 1, con picco a metà range
                normalized = dx / max_allowed
                horizontal_score = 1.0 - abs(0.5 - normalized) * 2
        
        # 2. Allineamento verticale (vuole dy piccolo)
        vertical_score = max(0, 1.0 - (dy / self.max_vertical_offset))
        
        # 3. Similarità area (vuole ratio alto)
        area_score = area_ratio
        
        # 4. Circolarità media (preferisce fari più circolari)
        circularity_score = (c1.circularity + c2.circularity) / 2
        
        # Score totale pesato
        weights = {
            'horizontal': 0.35,
            'vertical': 0.30,
            'area': 0.25,
            'circularity': 0.10
        }
        
        total_score = (
            weights['horizontal'] * horizontal_score +
            weights['vertical'] * vertical_score +
            weights['area'] * area_score +
            weights['circularity'] * circularity_score
        )
        
        metrics = {
            'dx': dx,
            'dy': dy,
            'area_ratio': area_ratio,
            'horizontal_score': horizontal_score,
            'vertical_score': vertical_score,
            'area_score': area_score,
            'circularity_score': circularity_score,
            'total_score': total_score
        }
        
        return total_score, metrics
    
    def select_tail_light_pair(self, candidates: List[LightCandidate]) -> Optional[Tuple[LightCandidate, LightCandidate]]:
        """
        Seleziona la coppia di fari posteriori migliore.
        
        Args:
            candidates: Lista di candidati rilevati
            
        Returns:
            Tuple (left_light, right_light) ordinati da sinistra a destra,
            oppure None se nessuna coppia valida
        """


        # FILTRO AGGIUNTIVO: rimuovi candidati troppo in alto o ai bordi
        frame_height = 1080  # Adatta alla tua risoluzione
        filtered_candidates = []
        
        for candidate in candidates:
            cx, cy = candidate.center
            
            # Scarta candidati:
            # - Troppo in alto (sopra metà frame)
            # - Troppo ai bordi (primi/ultimi 10% larghezza)
            # - Area troppo piccola o troppo grande
            if (cy > frame_height * 0.3 and  # Almeno nel terzo inferiore
                50 < cx < self.frame_width - 50 and  # Non ai bordi
                candidate.area > 100 and  # Area minima ragionevole
                candidate.circularity > 0.5):  # Abbastanza circolare
                filtered_candidates.append(candidate)
        
        candidates = filtered_candidates
        
        if len(candidates) < 2:
            return None
    

        if len(candidates) < 2:
            return None
        
        best_pair = None
        best_score = self.min_pair_score
        best_metrics = None
        
        # Prova tutte le combinazioni
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, c2 = candidates[i], candidates[j]
                
                score, metrics = self.compute_pair_score(c1, c2)
                
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    
                    # Ordina da sinistra a destra
                    if c1.center[0] < c2.center[0]:
                        best_pair = (c1, c2)
                    else:
                        best_pair = (c2, c1)
        
        if best_pair and best_metrics:
            print(f"✓ Coppia selezionata - Score: {best_score:.3f}")
            print(f"  dx={best_metrics['dx']:.0f}px, dy={best_metrics['dy']:.0f}px, "
                  f"area_ratio={best_metrics['area_ratio']:.2f}")
        
        return best_pair
    
    def get_tail_light_centers(self, candidates: List[LightCandidate]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Ottiene i centri della coppia di fari selezionata.
        
        Args:
            candidates: Lista di candidati
            
        Returns:
            Tuple ((left_x, left_y), (right_x, right_y)) oppure None
        """
        pair = self.select_tail_light_pair(candidates)
        
        if pair is None:
            return None
        
        left_light, right_light = pair
        return (left_light.center, right_light.center)
    
    def filter_by_previous_position(self, candidates: List[LightCandidate],
                                    previous_centers: Tuple[Tuple[int, int], Tuple[int, int]],
                                    max_distance: int = 100) -> List[LightCandidate]:
        """
        Filtra i candidati in base alla posizione precedente (per tracking).
        
        Args:
            candidates: Lista di candidati correnti
            previous_centers: Centri precedenti ((left_x, left_y), (right_x, right_y))
            max_distance: Distanza massima in pixel dalla posizione precedente
            
        Returns:
            Lista di candidati filtrati
        """
        if not candidates or not previous_centers:
            return candidates
        
        prev_left, prev_right = previous_centers
        
        filtered = []
        
        for candidate in candidates:
            cx, cy = candidate.center
            
            # Calcola distanza da entrambi i fari precedenti
            dist_left = np.sqrt((cx - prev_left[0])**2 + (cy - prev_left[1])**2)
            dist_right = np.sqrt((cx - prev_right[0])**2 + (cy - prev_right[1])**2)
            
            # Accetta se vicino a uno dei due fari precedenti
            if min(dist_left, dist_right) < max_distance:
                filtered.append(candidate)
        
        return filtered
    
    def estimate_missing_light(self, single_light: LightCandidate,
                              previous_centers: Tuple[Tuple[int, int], Tuple[int, int]],
                              is_left: bool) -> Tuple[int, int]:
        """
        Stima la posizione di un faro mancante basandosi sulla distanza precedente.
        
        Args:
            single_light: Faro rilevato
            previous_centers: Centri precedenti
            is_left: True se single_light è il faro sinistro
            
        Returns:
            Centro stimato del faro mancante (x, y)
        """
        prev_left, prev_right = previous_centers
        prev_distance = prev_right[0] - prev_left[0]
        
        if is_left:
            # Stima il faro destro
            estimated_x = single_light.center[0] + prev_distance
            estimated_y = single_light.center[1]
        else:
            # Stima il faro sinistro
            estimated_x = single_light.center[0] - prev_distance
            estimated_y = single_light.center[1]
        
        return (int(estimated_x), int(estimated_y))