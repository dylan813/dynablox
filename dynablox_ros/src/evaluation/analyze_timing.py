#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path


def read_dynablox_timings(timings_file):
    """Parse Dynablox timings.txt file (Voxblox format)."""
    timings = {}
    
    if not os.path.exists(timings_file):
        print(f"Warning: {timings_file} not found")
        return timings
    
    with open(timings_file, 'r') as f:
        lines = f.readlines()
        
    # Skip header lines
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) < 5:
            continue
            
        name = parts[0].strip()
        calls = int(parts[1])
        total = float(parts[2])
        
        # Parse mean ± stddev
        mean_std = parts[3].strip()
        mean_str = mean_std.split('(')[1].split('+')[0].strip()
        std_str = mean_std.split('+-')[1].split(')')[0].strip()
        mean = float(mean_str)
        std = float(std_str)
        
        # Parse [min, max]
        min_max = parts[4].strip()
        min_str = min_max.split('[')[1].split(',')[0].strip()
        max_str = min_max.split(',')[1].split(']')[0].strip()
        min_val = float(min_str)
        max_val = float(max_str)
        
        timings[name] = {
            'calls': calls,
            'total': total,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val
        }
    
    return timings


def read_pointosr_timings(timings_file):
    """Parse PointOSR timings file."""
    timings = {}
    
    if not os.path.exists(timings_file):
        print(f"Warning: {timings_file} not found")
        return timings
    
    with open(timings_file, 'r') as f:
        lines = f.readlines()
    
    # Find the data section
    data_started = False
    for line in lines:
        line = line.strip()
        if not data_started:
            if line.startswith('-' * 30):  # Look for separator
                data_started = True
            continue
        
        if line.startswith('-'):  # End of data section
            break
        
        parts = line.split()
        if len(parts) < 6:
            continue
        
        name = parts[0]
        calls = int(parts[1])
        total = float(parts[2])
        mean = float(parts[3])
        std = float(parts[4])
        min_val = float(parts[5])
        max_val = float(parts[6])
        
        timings[name] = {
            'calls': calls,
            'total': total,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val
        }
    
    return timings


def read_classification_latencies(latency_file):
    """Parse classification latency statistics."""
    if not os.path.exists(latency_file):
        print(f"Warning: {latency_file} not found")
        return None
    
    stats = {}
    with open(latency_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Number of samples:'):
                stats['samples'] = int(line.split(':')[1].strip())
            elif line.startswith('Mean:'):
                # Extract ms value
                ms_val = line.split('ms')[0].split(':')[1].strip()
                stats['mean_ms'] = float(ms_val)
            elif line.startswith('StdDev:'):
                ms_val = line.split('ms')[0].split(':')[1].strip()
                stats['std_ms'] = float(ms_val)
            elif line.startswith('Min:'):
                ms_val = line.split('ms')[0].split(':')[1].strip()
                stats['min_ms'] = float(ms_val)
            elif line.startswith('Max:'):
                ms_val = line.split('ms')[0].split(':')[1].strip()
                stats['max_ms'] = float(ms_val)
    
    return stats if stats else None


def analyze_directory(directory):
    """Analyze all timing files in a directory."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {directory}")
    print(f"{'='*80}\n")
    
    # Read Dynablox timings
    dynablox_file = os.path.join(directory, 'timings.txt')
    dynablox_timings = read_dynablox_timings(dynablox_file)
    
    # Read PointOSR timings
    pointosr_file = os.path.join(directory, 'pointosr_timings.txt')
    pointosr_timings = read_pointosr_timings(pointosr_file)
    
    # Read classification latencies
    latency_file = os.path.join(directory, 'classification_latencies.txt')
    latencies = read_classification_latencies(latency_file)
    
    # Display results
    has_pointosr = bool(pointosr_timings) or latencies
    
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print("│                        DYNABLOX TIMING SUMMARY                          │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    
    if 'frame' in dynablox_timings:
        frame = dynablox_timings['frame']
        print(f"\n  Overall Performance:")
        print(f"    Frames Processed:  {frame['calls']}")
        print(f"    Total Time:        {frame['total']:.2f} s")
        print(f"    Time per Frame:    {frame['mean']*1000:.1f} ± {frame['std']*1000:.1f} ms")
        print(f"    Frame Rate:        {frame['calls']/frame['total']:.1f} Hz")
        print(f"    Range:             [{frame['min']*1000:.1f}, {frame['max']*1000:.1f}] ms")
    
    print(f"\n  Component Breakdown:")
    print(f"    {'Component':<35} {'Mean (ms)':<12} {'%':<8} {'Calls':<8}")
    print(f"    {'-'*35} {'-'*12} {'-'*8} {'-'*8}")
    
    # Calculate total for percentages
    total_time = dynablox_timings.get('frame', {}).get('mean', 1.0)
    
    # Sort by mean time descending
    components = [
        ('TF Lookup', 'motion_detection/tf_lookup'),
        ('TSDF Integration', 'motion_detection/tsdf_integration'),
        ('Indexing Setup', 'motion_detection/indexing_setup'),
        ('Ever-Free Updates', 'motion_detection/update_ever_free'),
        ('Preprocessing', 'motion_detection/preprocessing'),
        ('Clustering', 'motion_detection/clustering'),
        ('Tracking', 'motion_detection/tracking'),
        ('Evaluation', 'evaluation'),
        ('Visualization', 'visualizations'),
    ]
    
    for display_name, key in components:
        if key in dynablox_timings:
            data = dynablox_timings[key]
            pct = (data['mean'] / total_time) * 100 if total_time > 0 else 0
            print(f"    {display_name:<35} {data['mean']*1000:>10.1f}   {pct:>6.1f}%  {data['calls']:<8}")
    
    # PointOSR section
    if has_pointosr:
        print("\n┌─────────────────────────────────────────────────────────────────────────┐")
        print("│                        POINTOSR TIMING SUMMARY                          │")
        print("└─────────────────────────────────────────────────────────────────────────┘")
        
        if pointosr_timings:
            print(f"\n  PointOSR Processing (per batch):")
            print(f"    {'Component':<35} {'Mean (ms)':<12} {'Calls':<8}")
            print(f"    {'-'*35} {'-'*12} {'-'*8}")
            
            for comp in ['data_conversion', 'model_inference', 'osr_scoring', 'result_packaging', 'total_batch_processing']:
                if comp in pointosr_timings:
                    data = pointosr_timings[comp]
                    display_name = comp.replace('_', ' ').title()
                    print(f"    {display_name:<35} {data['mean']*1000:>10.1f}   {data['calls']:<8}")
        
        if latencies:
            print(f"\n  End-to-End Classification Latency:")
            print(f"    Samples:           {latencies['samples']}")
            print(f"    Mean Latency:      {latencies['mean_ms']:.1f} ms")
            print(f"    Std Dev:           {latencies['std_ms']:.1f} ms")
            print(f"    Range:             [{latencies['min_ms']:.1f}, {latencies['max_ms']:.1f}] ms")
            print(f"\n    Note: This is the round-trip time from cluster publish to")
            print(f"          classification result received by Dynablox.")
    
    # Summary
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│                             SUMMARY                                     │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    
    if 'frame' in dynablox_timings:
        frame = dynablox_timings['frame']
        print(f"\n  System Throughput:     {frame['calls']/frame['total']:.1f} Hz")
        print(f"  Avg Frame Time:        {frame['mean']*1000:.1f} ms")
        
        if latencies:
            print(f"  Avg Classification Latency: {latencies['mean_ms']:.1f} ms")
            print(f"\n  The classification latency is additional time beyond frame")
            print(f"  processing for clusters to be classified and results returned.")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze timing metrics from Dynablox and PointOSR experiments'
    )
    parser.add_argument(
        'directories',
        nargs='+',
        help='One or more directories containing timing files (timings.txt, pointosr_timings.txt, etc.)'
    )
    
    args = parser.parse_args()
    
    for directory in args.directories:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory")
            continue
        
        analyze_directory(directory)


if __name__ == '__main__':
    main()

