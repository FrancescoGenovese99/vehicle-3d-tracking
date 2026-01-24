"""
Candidate Selector - Selezione della coppia di fari posteriori migliore.
Versione SEMPLIFICATA e ROBUSTA per Toyota Aygo X.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from .light_detector import LightCandidate


class CandidateSelector:
    """
    Seleziona la coppia di fari posteriori più probabile tra i candidati rilevati.
    
    STRATEGIA:
    1. Filtro SOFT (rimuove solo outliers evidenti)
    2. Score basato su: distanza orizzontale + allineamento verticale + intensità
    3. Preferenza per fari GRANDI (vicini) e LUMINOSI
    """
    
    def __init__(self, config: Dict, frame_width: int, frame_height: int = 1080):
        """
        Inizializza il selettore.
        
        Args:
            config: Dizionario di configurazione (da detection_params.yaml)
            frame_width: Larghezza del frame in pixel
            frame_height: Altezza del frame in pixel
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Parametri di selezione
        sel_cfg = config.get('tail_lights_selection', {})
        self.min_horizontal_distance = sel_cfg.get('min_horizontal_distance', 50)
        self.max_horizontal_distance_ratio = sel_cfg.get('max_horizontal_distance_ratio', 0.8)
        self.max_vertical_offset = sel_cfg.get('max_vertical_offset', 60)
        self.min_area_similarity = sel_cfg.get('min_area_similarity', 0.4)
        self.min_pair_score = sel_cfg.get('min_pair_score', 0.25)
        
        print(f"[CandidateSelector] Init: {frame_width}x{frame_height}")
        print(f"  Min H-distance: {self.min_horizontal_distance}px")
        print(f"  Max V-offset: {self.max_vertical_offset}px")
    
    def soft_filter_candidates(self, candidates: List[LightCandidate]) -> List[LightCandidate]:
        """
        Filtro SOFT: rimuove solo outliers evidenti.
        
        Criteri MINIMI:
        - Non nei bordi estremi (margine 30px)
        - Area > 50 px² (molto permissivo)
        - Circolarità > 0.05 (accetta forme allungate)
        - Non troppo vicino al bordo superiore (primi 10%)
        - Non troppo vicino al bordo inferiore (ultimi 5%)
        """
        filtered = []
        
        for c in candidates:
            cx, cy = c.center
            
            # Filtro 1: Non ai bordi estremi
            if cx < 30 or cx > self.frame_width - 30:
                continue
            
            # Filtro 2: Zona Y ragionevole (10%-95% del frame)
            y_ratio = cy / self.frame_height
            if y_ratio < 0.10 or y_ratio > 0.95:
                continue
            
            # Filtro 3: Area minima (molto permissiva)
            if c.area < 50:
                continue
            
            # Filtro 4: Forma non troppo irregolare
            if c.circularity < 0.05:
                continue
            
            filtered.append(c)
        
        return filtered
    
    def compute_pair_score(self, c1: LightCandidate, c2: LightCandidate) -> Tuple[float, Dict]:
        """
        Calcola score per coppia di candidati.
        
        STRATEGIA ADATTIVA:
        1. SIZE IMPORTANTE ma non dominante (peso 35%)
        2. Allineamento verticale CRITICO (peso 35%)
        3. Distanza orizzontale (peso 20%)
        4. Similarità area (peso 10%)
        
        Questo funziona ANCHE quando il veicolo si allontana (area diminuisce).
        """
        dx = abs(c1.center[0] - c2.center[0])
        dy = abs(c1.center[1] - c2.center[1])
        area_ratio = min(c1.area, c2.area) / max(c1.area, c2.area)
        avg_area = (c1.area + c2.area) / 2
        
        # ========== SCORE 1: SIZE SCORE (ADATTIVO) ==========
        # Fari vicini: 400-800px²
        # Fari medi: 150-400px²
        # Fari lontani: 80-150px²
        # Auto parcheggiate illuminate: 100-150px²
        
        size_score = 0.0
        if avg_area >= 400:
            # Fari ENORMI (molto vicini) → Score MASSIMO
            size_score = 1.0
        elif avg_area >= 200:
            # Fari grandi-medi (vicini-medi) → Score alto
            size_score = 0.7 + (avg_area - 200) / 200 * 0.3
        elif avg_area >= 100:
            # Fari medi-piccoli (medi-lontani) → Score medio
            size_score = 0.4 + (avg_area - 100) / 100 * 0.3
        else:
            # Fari molto piccoli (molto lontani o outliers) → Score basso
            size_score = max(0.1, avg_area / 100 * 0.3)
        
        # ========== SCORE 2: Allineamento Verticale (CRITICO) ==========
        # QUESTO È IL DISCRIMINANTE PRINCIPALE per fari reali vs outliers
        vertical_score = max(0, 1.0 - (dy / self.max_vertical_offset))
        
        # ========== SCORE 3: Distanza Orizzontale (ADATTIVA) ==========
        # Fari vicini: 200-400px (ben distanziati)
        # Fari lontani: 50-150px (più ravvicinati)
        
        horizontal_score = 0.0
        if self.min_horizontal_distance <= dx <= self.frame_width * self.max_horizontal_distance_ratio:
            if dx >= 200:
                # Fari VICINI (ben distanziati) → Score molto alto
                horizontal_score = min(1.0, 0.8 + (dx - 200) / 300 * 0.2)
            elif dx >= 100:
                # Fari MEDI (distanza media) → Score alto
                horizontal_score = 0.6 + (dx - 100) / 100 * 0.2
            else:
                # Fari LONTANI (più ravvicinati) → Score medio
                horizontal_score = 0.4 + (dx - self.min_horizontal_distance) / 50 * 0.2
        
        # ========== SCORE 4: Similarità Area ==========
        area_score = area_ratio
        
        # ========== WEIGHTED TOTAL (BILANCIATO) ==========
        weights = {
            'size': 0.35,          # SIZE importante ma non dominante
            'vertical': 0.35,      # ALLINEAMENTO CRITICO (discrimina outliers)
            'horizontal': 0.20,    # DISTANZA importante
            'area': 0.10          # SIMILARITÀ meno critica
        }
        
        total_score = (
            weights['size'] * size_score +
            weights['vertical'] * vertical_score +
            weights['horizontal'] * horizontal_score +
            weights['area'] * area_score
        )
        
        metrics = {
            'dx': dx,
            'dy': dy,
            'area_ratio': area_ratio,
            'avg_area': avg_area,
            'size_score': size_score,           # PRIORITÀ 1
            'vertical_score': vertical_score,
            'horizontal_score': horizontal_score,
            'area_score': area_score,
            'total_score': total_score
        }
        
        return total_score, metrics
    
    def select_tail_light_pair(self, candidates: List[LightCandidate], prefer_center: bool = False) -> Optional[Tuple[LightCandidate, LightCandidate]]:
        """
        Seleziona la coppia di fari posteriori migliore.
        
        Pipeline:
        1. Filtro SOFT (rimuove solo outliers evidenti)
        2. Score tutte le combinazioni
        3. [OPZIONALE] Bonus per fari vicini al centro frame (detection iniziale)
        4. Ritorna coppia con score massimo
        
        Args:
            candidates: Lista di candidati rilevati
            prefer_center: Se True, aggiunge bonus per fari vicini al centro orizzontale
                          (utile per detection iniziale quando veicolo entra nel frame)
        """
        # Step 1: Filtro soft
        candidates = self.soft_filter_candidates(candidates)
        
        if len(candidates) < 2:
            return None
        
        # Step 2: Score tutte le coppie
        best_pair = None
        best_score = self.min_pair_score
        best_metrics = None
        
        center_x = self.frame_width / 2
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, c2 = candidates[i], candidates[j]
                
                score, metrics = self.compute_pair_score(c1, c2)
                
                # Step 3: Bonus per fari centrali (detection iniziale)
                if prefer_center:
                    pair_center_x = (c1.center[0] + c2.center[0]) / 2
                    distance_from_center = abs(pair_center_x - center_x) / self.frame_width
                    center_bonus = (1.0 - distance_from_center) * 0.15  # Max +15% bonus
                    score += center_bonus
                
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    
                    # Ordina da sinistra a destra
                    if c1.center[0] < c2.center[0]:
                        best_pair = (c1, c2)
                    else:
                        best_pair = (c2, c1)
        
        # Step 3: Log risultato
        if best_pair and best_metrics:
            left, right = best_pair
            y_avg = (left.center[1] + right.center[1]) / 2
            y_percent = (y_avg / self.frame_height) * 100
            
            # Classificazione distanza basata su area
            if best_metrics['avg_area'] >= 400:
                distance_class = "VERY CLOSE"
            elif best_metrics['avg_area'] >= 200:
                distance_class = "CLOSE-MEDIUM"
            elif best_metrics['avg_area'] >= 100:
                distance_class = "MEDIUM-FAR"
            else:
                distance_class = "FAR"
            
            print(f"   ✓ Pair selected - Score: {best_score:.3f} ({distance_class})")
            print(f"     dx={best_metrics['dx']:.0f}px, dy={best_metrics['dy']:.0f}px")
            print(f"     Area: {best_metrics['avg_area']:.0f}px², Y: {y_percent:.1f}% frame")
            print(f"     Scores: SIZE={best_metrics['size_score']:.2f} (35%), "
                  f"V={best_metrics['vertical_score']:.2f} (35%), "
                  f"H={best_metrics['horizontal_score']:.2f} (20%)")
        
        return best_pair
    
    def get_tail_light_centers(self, candidates: List[LightCandidate], prefer_center: bool = False) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Ottiene i centri della coppia di fari selezionata.
        
        Args:
            candidates: Lista di candidati rilevati
            prefer_center: Se True, preferisce fari vicini al centro frame
        """
        pair = self.select_tail_light_pair(candidates, prefer_center=prefer_center)
        
        if pair is None:
            return None
        
        left_light, right_light = pair
        return (left_light.center, right_light.center)
    
    def filter_by_previous_position(self, candidates: List[LightCandidate],
                                    previous_centers: Tuple[Tuple[int, int], Tuple[int, int]],
                                    max_distance: int = 150) -> List[LightCandidate]:
        """
        Filtra candidati vicini alla posizione precedente (per tracking).
        """
        if not candidates or not previous_centers:
            return candidates
        
        prev_left, prev_right = previous_centers
        filtered = []
        
        for candidate in candidates:
            cx, cy = candidate.center
            
            # Distanza da entrambi i fari precedenti
            dist_left = np.sqrt((cx - prev_left[0])**2 + (cy - prev_left[1])**2)
            dist_right = np.sqrt((cx - prev_right[0])**2 + (cy - prev_right[1])**2)
            
            # Accetta se vicino a uno dei due
            if min(dist_left, dist_right) < max_distance:
                filtered.append(candidate)
        
        return filtered
    
    def estimate_missing_light(self, single_light: LightCandidate,
                              previous_centers: Tuple[Tuple[int, int], Tuple[int, int]],
                              is_left: bool) -> Tuple[int, int]:
        """
        Stima posizione faro mancante basandosi sulla distanza precedente.
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