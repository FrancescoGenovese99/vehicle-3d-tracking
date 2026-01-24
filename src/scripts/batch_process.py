#!/usr/bin/env python3
"""
Script per processare in batch tutti i video in una directory.

Usage:
    python scripts/batch_process.py --input-dir data/videos/input \
                                     --output-dir data/videos/output \
                                     --save-results \
                                     --parallel 2
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List
import time

# Aggiungi la directory root al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Processa in batch tutti i video in una directory',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/videos/input',
        help='Directory contenente i video di input'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/videos/output',
        help='Directory per i video di output'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mp4',
        help='Pattern per i file video (default: *.mp4)'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Salva risultati numerici per ogni video'
    )
    
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        default=True,
        help='Disegna bounding box 3D'
    )
    
    parser.add_argument(
        '--draw-axes',
        action='store_true',
        help='Disegna assi del sistema di riferimento'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Numero di video da processare in parallelo (default: 1)'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continua il batch anche se un video fallisce'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostra cosa verrebbe processato senza eseguire'
    )
    
    return parser.parse_args()


def find_videos(input_dir: str, pattern: str) -> List[Path]:
    """
    Trova tutti i video nella directory.
    
    Args:
        input_dir: Directory di input
        pattern: Pattern per i file
        
    Returns:
        Lista di Path ai video
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Directory non trovata: {input_dir}")
    
    videos = list(input_path.glob(pattern))
    
    # Supporta anche altri formati comuni
    if pattern == '*.mp4':
        videos.extend(input_path.glob('*.avi'))
        videos.extend(input_path.glob('*.mov'))
        videos.extend(input_path.glob('*.MP4'))
        videos.extend(input_path.glob('*.AVI'))
        videos.extend(input_path.glob('*.MOV'))
    
    return sorted(videos)


def process_single_video(video_path: Path, output_dir: str, args) -> int:
    """
    Processa un singolo video chiamando process_video.py.
    
    Args:
        video_path: Path del video
        output_dir: Directory di output
        args: Argomenti da command line
        
    Returns:
        Return code del subprocess
    """
    output_path = Path(output_dir) / f"{video_path.stem}_tracked.mp4"
    
    # Costruisci comando
    cmd = [
        'python',
        'scripts/process_video.py',
        '--input', str(video_path),
        '--output', str(output_path)
    ]
    
    if args.save_results:
        cmd.append('--save-results')
    
    if args.draw_bbox:
        cmd.append('--draw-bbox')
    
    if args.draw_axes:
        cmd.append('--draw-axes')
    
    print(f"\n{'='*60}")
    print(f"Processando: {video_path.name}")
    print(f"Output: {output_path.name}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print(f"[DRY-RUN] Comando: {' '.join(cmd)}")
        return 0
    
    # Esegui
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Completato in {elapsed:.1f}s: {video_path.name}")
    else:
        print(f"\n✗ Fallito dopo {elapsed:.1f}s: {video_path.name}")
    
    return result.returncode


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("BATCH VIDEO PROCESSING")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Save results: {args.save_results}")
    print(f"Parallel jobs: {args.parallel}")
    print("="*60)
    
    # Trova video
    try:
        videos = find_videos(args.input_dir, args.pattern)
    except FileNotFoundError as e:
        print(f"\n❌ ERRORE: {e}")
        return 1
    
    if not videos:
        print(f"\n❌ Nessun video trovato in {args.input_dir} con pattern {args.pattern}")
        return 1
    
    print(f"\nTrovati {len(videos)} video da processare:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")
    
    if args.dry_run:
        print("\n[DRY-RUN] Nessun video verrà processato")
    
    # Crea output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Processa video
    results = []
    failed = []
    
    if args.parallel > 1:
        print(f"\n⚠️  Parallel processing non ancora implementato, usando sequential")
        print("   (verrà aggiunto in futuro con multiprocessing)")
    
    # Sequential processing
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] ", end='')
        
        try:
            returncode = process_single_video(video, args.output_dir, args)
            results.append((video, returncode))
            
            if returncode != 0:
                failed.append(video)
                if not args.continue_on_error:
                    print(f"\n❌ Interrotto per errore in: {video.name}")
                    break
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrotto dall'utente")
            break
        except Exception as e:
            print(f"\n❌ Errore inaspettato processando {video.name}: {e}")
            failed.append(video)
            if not args.continue_on_error:
                break
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETATO")
    print("="*60)
    
    successful = len([r for r in results if r[1] == 0])
    print(f"\nRisultati:")
    print(f"  Totale video: {len(videos)}")
    print(f"  Processati: {len(results)}")
    print(f"  Successo: {successful}")
    print(f"  Falliti: {len(failed)}")
    
    if failed:
        print(f"\nVideo falliti:")
        for video in failed:
            print(f"  - {video.name}")
    
    print("="*60)
    
    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())