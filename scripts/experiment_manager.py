#!/usr/bin/env python3
"""
Helper script to list, monitor, and manage experiments.

Usage:
    python scripts/experiment_manager.py --list          # List running experiments
    python scripts/experiment_manager.py --status       # Show status of all experiments
    python scripts/experiment_manager.py --kill SESSION # Kill a specific tmux session
    python scripts/experiment_manager.py --logs SESSION # Tail logs for a session
"""

import argparse
import subprocess
import os
import glob
import json
from datetime import datetime

def list_tmux_sessions():
    """List all tmux sessions."""
    try:
        result = subprocess.run(['tmux', 'list-sessions'], 
                              capture_output=True, text=True, check=True)
        sessions = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(':')
                if len(parts) >= 2:
                    sessions.append(parts[0])
        return sessions
    except subprocess.CalledProcessError:
        return []

def get_experiment_status():
    """Get status of all experiments."""
    sessions = list_tmux_sessions()
    experiment_sessions = [s for s in sessions if 'experiment' in s.lower()]
    
    status = {}
    for session in experiment_sessions:
        status[session] = {
            'session': session,
            'running': True,
            'log_file': None,
            'results_dir': None
        }
        
        # Try to find log file
        log_pattern = f"results/*/logs/*{session.split('_')[0]}*.log"
        log_files = glob.glob(log_pattern)
        if log_files:
            status[session]['log_file'] = log_files[0]
            status[session]['results_dir'] = os.path.dirname(os.path.dirname(log_files[0]))
    
    return status

def tail_logs(session_name):
    """Tail logs for a specific session."""
    status = get_experiment_status()
    if session_name in status and status[session_name]['log_file']:
        log_file = status[session_name]['log_file']
        print(f"Tailing log file: {log_file}")
        os.system(f"tail -f {log_file}")
    else:
        print(f"No log file found for session: {session_name}")

def kill_session(session_name):
    """Kill a tmux session."""
    try:
        subprocess.run(['tmux', 'kill-session', '-t', session_name], check=True)
        print(f"Killed session: {session_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill session {session_name}: {e}")

def show_experiment_results():
    """Show summary of experiment results."""
    results_dirs = glob.glob("results/experiment*")
    
    print("Experiment Results Summary:")
    print("=" * 50)
    
    for results_dir in sorted(results_dirs):
        print(f"\n{results_dir}:")
        
        # Find CSV files
        csv_files = glob.glob(f"{results_dir}/*/experiment_results.csv")
        if csv_files:
            for csv_file in csv_files:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    print(f"  CSV: {csv_file}")
                    print(f"    Experiments: {len(df)}")
                    if 'test_loss' in df.columns:
                        print(f"    Best test loss: {df['test_loss'].min():.6f}")
                        print(f"    Mean test loss: {df['test_loss'].mean():.6f}")
                except Exception as e:
                    print(f"    Error reading CSV: {e}")
        else:
            print("  No CSV results found")

def main():
    parser = argparse.ArgumentParser(description='Manage neural network experiments')
    parser.add_argument('--list', action='store_true',
                       help='List running tmux sessions')
    parser.add_argument('--status', action='store_true',
                       help='Show status of all experiments')
    parser.add_argument('--kill', type=str,
                       help='Kill specific tmux session')
    parser.add_argument('--logs', type=str,
                       help='Tail logs for specific session')
    parser.add_argument('--results', action='store_true',
                       help='Show summary of experiment results')
    
    args = parser.parse_args()
    
    if args.list:
        sessions = list_tmux_sessions()
        print("Running tmux sessions:")
        for session in sessions:
            print(f"  {session}")
    
    elif args.status:
        status = get_experiment_status()
        print("Experiment Status:")
        print("=" * 50)
        for session, info in status.items():
            print(f"\nSession: {session}")
            print(f"  Running: {info['running']}")
            print(f"  Log file: {info['log_file']}")
            print(f"  Results: {info['results_dir']}")
    
    elif args.kill:
        kill_session(args.kill)
    
    elif args.logs:
        tail_logs(args.logs)
    
    elif args.results:
        show_experiment_results()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
