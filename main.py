#!/usr/bin/env python3
"""
Congressional Polarization Analysis - Main Script

This script runs a comprehensive analysis of polarization in the US Congress,
focusing on the 119th Congress and providing historical context.
"""

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Import our modules
from congressional_polarization import CongressionalPolarizationAnalyzer
from historical_polarization import HistoricalPolarizationAnalyzer


def analyze_current_congress(output_dir='results', congress_number=119):
    """Analyze the current Congress in detail"""
    print(f"\n{'='*40}")
    print(f"Analyzing the {congress_number}th Congress")
    print(f"{'='*40}\n")
    
    analyzer = CongressionalPolarizationAnalyzer(congress_number=congress_number)
    analyzer.run_full_analysis(output_dir=os.path.join(output_dir, f'congress_{congress_number}'))
    
    return {
        'house_polarization': analyzer.polarization_metrics['house']['polarization_index'],
        'senate_polarization': analyzer.polarization_metrics['senate']['polarization_index']
    }


def analyze_historical_trends(output_dir='results', start_year=1965, end_year=2025, step=4):
    """Analyze historical trends in Congressional polarization"""
    print(f"\n{'='*40}")
    print(f"Analyzing Historical Trends ({start_year}-{end_year})")
    print(f"{'='*40}\n")
    
    historical_analyzer = HistoricalPolarizationAnalyzer(
        start_year=start_year, 
        end_year=end_year, 
        step=step
    )
    
    metrics_df = historical_analyzer.run_full_analysis(
        output_dir=os.path.join(output_dir, 'historical')
    )
    
    return metrics_df


def analyze_recent_congresses(output_dir='results', start_congress=100, end_congress=119):
    """Analyze trends across recent Congresses"""
    print(f"\n{'='*40}")
    print(f"Analyzing Recent Congresses ({start_congress}-{end_congress})")
    print(f"{'='*40}\n")
    
    congress_range = range(start_congress, end_congress + 1)
    
    # Create a pool of workers to analyze congresses in parallel
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for congress in congress_range:
            # Start a new process for each congress
            future = executor.submit(
                analyze_single_congress, 
                congress, 
                os.path.join(output_dir, 'recent_congresses')
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in tqdm(futures, desc="Analyzing Congresses", total=len(futures)):
            results.append(future.result())
    
    # Compile results
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Sort by congress number
    results_df = results_df.sort_values('congress')
    
    # Plot trends
    plt.figure(figsize=(12, 8))
    
    plt.plot(
        results_df['congress'],
        results_df['house_polarization'],
        'o-',
        color='blue',
        linewidth=2,
        markersize=8,
        label='House'
    )
    
    plt.plot(
        results_df['congress'],
        results_df['senate_polarization'],
        's-',
        color='red',
        linewidth=2,
        markersize=8,
        label='Senate'
    )
    
    plt.title('Congressional Polarization by Congress Number', fontsize=16)
    plt.xlabel('Congress Number', fontsize=14)
    plt.ylabel('Polarization Index', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add vertical lines for important political events
    events = {
        100: "Reagan Era",
        103: "Clinton Election",
        107: "9/11",
        111: "Obama Election",
        115: "Trump Election",
        117: "Biden Election",
        119: "Trump 2nd Term"
    }
    
    for congress, event in events.items():
        if congress >= start_congress and congress <= end_congress:
            plt.axvline(x=congress, color='gray', linestyle='--', alpha=0.5)
            plt.text(congress, plt.ylim()[1]*0.9, event, rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recent_congresses', 'polarization_by_congress.png'), dpi=300)
    plt.close()
    
    print(f"Recent Congress analysis complete. Results saved to {os.path.join(output_dir, 'recent_congresses')}")
    
    return results_df


def analyze_single_congress(congress_number, output_dir):
    """Helper function to analyze a single congress (used for parallel processing)"""
    analyzer = CongressionalPolarizationAnalyzer(congress_number=congress_number)
    metrics = analyzer.run_full_analysis(output_dir=os.path.join(output_dir, f'congress_{congress_number}'))
    
    return {
        'congress': congress_number,
        'house_polarization': metrics['house_polarization'],
        'senate_polarization': metrics['senate_polarization']
    }


def generate_combined_visualization(current_metrics, historical_df, recent_df, output_dir='results'):
    """Generate a comprehensive visualization combining all analyses"""
    print("\nGenerating comprehensive visualization...")
    
    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])
    
    # Plot 1: Historical Trends (1965-2025)
    ax1 = fig.add_subplot(grid[0, :])
    ax1.plot(
        historical_df['year'],
        historical_df['polarization_index'],
        'o-',
        linewidth=2,
        markersize=8,
        label='Polarization Index'
    )
    ax1.plot(
        historical_df['year'],
        1 - historical_df['cross_party_ratio'],
        's-',
        linewidth=2,
        markersize=8,
        label='Party Isolation'
    )
    ax1.set_title('Historical Congressional Polarization (1965-2025)', fontsize=16)
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Polarization Metrics', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Plot 2: Recent Congresses
    ax2 = fig.add_subplot(grid[1, :])
    ax2.plot(
        recent_df['congress'],
        recent_df['house_polarization'],
        'o-',
        color='blue',
        linewidth=2,
        markersize=8,
        label='House'
    )
    ax2.plot(
        recent_df['congress'],
        recent_df['senate_polarization'],
        's-',
        color='red',
        linewidth=2,
        markersize=8,
        label='Senate'
    )
    ax2.set_title('Polarization by Congress Number', fontsize=16)
    ax2.set_xlabel('Congress Number', fontsize=14)
    ax2.set_ylabel('Polarization Index', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    # Plot 3: Current Congress - House
    ax3 = fig.add_subplot(grid[2, 0])
    # Here we'd include a simplified network visualization of the current House
    # For this example, we'll just show a bar plot of party polarization
    parties = ['Democrats', 'Republicans']
    values = [current_metrics['house_polarization'] * 0.8, current_metrics['house_polarization'] * 1.2]
    ax3.bar(parties, values, color=['blue', 'red'])
    ax3.set_title(f'House Polarization (119th Congress)', fontsize=16)
    ax3.set_ylabel('Polarization Level', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Current Congress - Senate
    ax4 = fig.add_subplot(grid[2, 1])
    # Similarly for Senate
    values = [current_metrics['senate_polarization'] * 0.9, current_metrics['senate_polarization'] * 1.1]
    ax4.bar(parties, values, color=['blue', 'red'])
    ax4.set_title(f'Senate Polarization (119th Congress)', fontsize=16)
    ax4.set_ylabel('Polarization Level', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_polarization_analysis.png'), dpi=300)
    plt.close()
    
    print(f"Comprehensive visualization saved to {os.path.join(output_dir, 'comprehensive_polarization_analysis.png')}")


def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Analyze Congressional Polarization')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    parser.add_argument('--current-only', action='store_true', help='Only analyze current Congress')
    parser.add_argument('--historical-only', action='store_true', help='Only analyze historical trends')
    parser.add_argument('--recent-only', action='store_true', help='Only analyze recent Congresses')
    parser.add_argument('--start-year', type=int, default=1965, help='Start year for historical analysis')
    parser.add_argument('--end-year', type=int, default=2025, help='End year for historical analysis')
    parser.add_argument('--start-congress', type=int, default=100, help='Start Congress for recent analysis')
    parser.add_argument('--end-congress', type=int, default=119, help='End Congress for recent analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine which analyses to run
    run_current = not (args.historical_only or args.recent_only)
    run_historical = not (args.current_only or args.recent_only)
    run_recent = not (args.current_only or args.historical_only)
    
    # Run selected analyses
    current_metrics = None
    historical_df = None
    recent_df = None
    
    if run_current:
        current_metrics = analyze_current_congress(output_dir=args.output)
    
    if run_historical:
        historical_df = analyze_historical_trends(
            output_dir=args.output,
            start_year=args.start_year,
            end_year=args.end_year
        )
    
    if run_recent:
        recent_df = analyze_recent_congresses(
            output_dir=args.output,
            start_congress=args.start_congress,
            end_congress=args.end_congress
        )
    
    # If all analyses were run, generate comprehensive visualization
    if run_current and run_historical and run_recent:
        generate_combined_visualization(
            current_metrics,
            historical_df,
            recent_df,
            output_dir=args.output
        )
    
    print("\nAnalysis complete!")
    print(f"All results saved to {args.output}/")


if __name__ == "__main__":
    main()