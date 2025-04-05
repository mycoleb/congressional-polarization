#!/usr/bin/env python3
"""
Historical Congressional Polarization Analysis

This module analyzes congressional polarization across a range of years,
creating visualizations similar to the example showing increasing polarization over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


class HistoricalPolarizationAnalyzer:
    def __init__(self, start_year=1965, end_year=2025, step=4):
        """
        Analyze congressional polarization across a range of years
        
        Parameters:
        -----------
        start_year : int
            Starting year for analysis (default: 1965, similar to example image)
        end_year : int
            Ending year for analysis (default: 2025, current)
        step : int
            Number of years between each analysis (default: 4)
        """
        self.years = list(range(start_year, end_year + 1, step))
        self.networks = {}
        
    def generate_historical_networks(self):
        """
        Generate simulated historical congressional networks with increasing polarization
        """
        print("Generating historical networks...")
        
        for year_idx, year in enumerate(self.years):
            # Create network
            G = nx.Graph()
            
            # Number of members (will use House size of ~435)
            n_members = 435
            
            # Create parameters that change over time to simulate increased polarization
            # Later years will have more polarization
            polarization_factor = min(0.2 + 0.7 * (year_idx / (len(self.years) - 1)), 0.9)
            
            # Ratio of parties (roughly equal with slight variations)
            dem_ratio = 0.5 + 0.05 * np.sin(year_idx * 0.5)
            
            for i in range(n_members):
                # Assign party based on ratio
                if i < n_members * dem_ratio:
                    party = 'D'
                else:
                    party = 'R'
                
                # Add node
                G.add_node(i, party=party)
            
            # Assign 2D positions to nodes based on party and polarization
            pos = {}
            
            for i in range(n_members):
                party = G.nodes[i]['party']
                
                # Generate positions with increasing separation between parties over time
                if party == 'D':
                    x = np.random.normal(-polarization_factor, 0.3 * (1 - polarization_factor))
                    y = np.random.normal(0, 0.4)
                else:  # 'R'
                    x = np.random.normal(polarization_factor, 0.3 * (1 - polarization_factor))
                    y = np.random.normal(0, 0.4)
                
                pos[i] = np.array([x, y])
            
            # Add connections based on proximity (closer nodes more likely to be connected)
            # Also more likely to connect within party, especially in later years
            for i in range(n_members):
                for j in range(i+1, n_members):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    
                    # Higher probability of edge if nodes are close
                    base_prob = max(0, 0.5 - dist)
                    
                    # Higher probability if same party, increasing with polarization
                    if G.nodes[i]['party'] == G.nodes[j]['party']:
                        prob = base_prob + 0.3 * polarization_factor
                    else:
                        prob = base_prob * (1 - 0.7 * polarization_factor)
                    
                    if np.random.random() < prob:
                        G.add_edge(i, j)
            
            self.networks[year] = (G, pos)
        
        print(f"Generated {len(self.years)} historical networks.")
    
    def visualize_historical_networks(self, output_dir='visualizations', grid_size=(4, 4)):
        """
        Visualize historical networks similar to the example image
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations (default: 'visualizations')
        grid_size : tuple
            Grid size for subplot layout (default: (4, 4))
        """
        if not self.networks:
            raise ValueError("Networks not generated. Run generate_historical_networks() first.")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate number of figures needed
        n_figures = (len(self.years) - 1) // (grid_size[0] * grid_size[1]) + 1
        
        for fig_idx in range(n_figures):
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(grid_size[0], grid_size[1], figure=fig)
            
            start_idx = fig_idx * grid_size[0] * grid_size[1]
            end_idx = min(start_idx + grid_size[0] * grid_size[1], len(self.years))
            
            for i, year_idx in enumerate(range(start_idx, end_idx)):
                year = self.years[year_idx]
                G, pos = self.networks[year]
                
                ax = fig.add_subplot(gs[i // grid_size[1], i % grid_size[1]])
                
                # Get node colors based on party
                node_colors = ['blue' if G.nodes[n]['party'] == 'D' else 'red' for n in G.nodes()]
                
                # Draw nodes
                nx.draw_networkx_nodes(
                    G, pos, 
                    node_size=30, 
                    node_color=node_colors, 
                    alpha=0.8,
                    ax=ax
                )
                
                # Draw edges with low alpha
                nx.draw_networkx_edges(
                    G, pos, 
                    width=0.5, 
                    alpha=0.1,
                    ax=ax
                )
                
                ax.set_title(f"{year}", fontsize=12)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/historical_networks_{fig_idx+1}.png", dpi=300)
            plt.close()
        
        print(f"Historical network visualizations saved to {output_dir}/")
    
    def calculate_polarization_metrics(self):
        """
        Calculate polarization metrics for each year
        """
        if not self.networks:
            raise ValueError("Networks not generated. Run generate_historical_networks() first.")
        
        print("Calculating polarization metrics...")
        
        metrics = []
        
        for year in self.years:
            G, pos = self.networks[year]
            
            # Calculate party-based metrics
            dem_nodes = [n for n in G.nodes() if G.nodes[n]['party'] == 'D']
            rep_nodes = [n for n in G.nodes() if G.nodes[n]['party'] == 'R']
            
            # 1. Cross-party edges ratio (lower means more polarized)
            total_edges = G.number_of_edges()
            cross_party_edges = sum(1 for u, v in G.edges() if G.nodes[u]['party'] != G.nodes[v]['party'])
            cross_party_ratio = cross_party_edges / total_edges if total_edges > 0 else 0
            
            # 2. Calculate distance between party centroids
            dem_positions = np.array([pos[n] for n in dem_nodes])
            rep_positions = np.array([pos[n] for n in rep_nodes])
            
            dem_centroid = dem_positions.mean(axis=0)
            rep_centroid = rep_positions.mean(axis=0)
            
            centroid_distance = np.linalg.norm(dem_centroid - rep_centroid)
            
            # 3. Within-party cohesion
            dem_cohesion = np.mean([np.linalg.norm(pos[n] - dem_centroid) for n in dem_nodes])
            rep_cohesion = np.mean([np.linalg.norm(pos[n] - rep_centroid) for n in rep_nodes])
            
            # 4. Polarization index
            avg_cohesion = (dem_cohesion + rep_cohesion) / 2
            polarization_index = centroid_distance / avg_cohesion if avg_cohesion > 0 else 0
            
            # 5. Cluster modularity (higher means more polarized)
            party_dict = {n: 0 if G.nodes[n]['party'] == 'D' else 1 for n in G.nodes()}
            try:
                modularity = nx.algorithms.community.modularity(G, [dem_nodes, rep_nodes])
            except:
                modularity = 0  # Fallback if calculation fails
            
            metrics.append({
                'year': year,
                'cross_party_ratio': cross_party_ratio,
                'centroid_distance': centroid_distance,
                'dem_cohesion': dem_cohesion,
                'rep_cohesion': rep_cohesion,
                'polarization_index': polarization_index,
                'modularity': modularity
            })
        
        self.metrics_df = pd.DataFrame(metrics)
        print("Polarization metrics calculated.")
        
        return self.metrics_df
    
    def visualize_polarization_trends(self, output_dir='visualizations'):
        """
        Visualize trends in polarization metrics over time
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations (default: 'visualizations')
        """
        if not hasattr(self, 'metrics_df'):
            raise ValueError("Metrics not calculated. Run calculate_polarization_metrics() first.")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot polarization trends
        plt.figure(figsize=(12, 8))
        
        plt.plot(
            self.metrics_df['year'],
            self.metrics_df['polarization_index'],
            'o-',
            linewidth=2,
            markersize=8,
            label='Polarization Index'
        )
        
        plt.plot(
            self.metrics_df['year'],
            1 - self.metrics_df['cross_party_ratio'],  # Convert to polarization measure (higher = more polarized)
            's-',
            linewidth=2,
            markersize=8,
            label='Party Isolation'
        )
        
        plt.plot(
            self.metrics_df['year'],
            self.metrics_df['modularity'],
            '^-',
            linewidth=2,
            markersize=8,
            label='Party Modularity'
        )
        
        plt.title('Congressional Polarization Trends (1965-2025)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Polarization Metrics', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/polarization_trends.png', dpi=300)
        plt.close()
        
        # Plot party cohesion
        plt.figure(figsize=(12, 8))
        
        plt.plot(
            self.metrics_df['year'],
            self.metrics_df['dem_cohesion'],
            'o-',
            color='blue',
            linewidth=2,
            markersize=8,
            label='Democratic Cohesion'
        )
        
        plt.plot(
            self.metrics_df['year'],
            self.metrics_df['rep_cohesion'],
            's-',
            color='red',
            linewidth=2,
            markersize=8,
            label='Republican Cohesion'
        )
        
        plt.title('Party Cohesion Trends (1965-2025)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Within-Party Cohesion', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/party_cohesion_trends.png', dpi=300)
        plt.close()
        
        print(f"Polarization trend visualizations saved to {output_dir}/")
    
    def run_full_analysis(self, output_dir='visualizations', grid_size=(4, 4)):
        """
        Run the full analysis pipeline
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations (default: 'visualizations')
        grid_size : tuple
            Grid size for subplot layout (default: (4, 4))
        """
        print(f"Running full historical analysis from {self.years[0]} to {self.years[-1]}...")
        
        self.generate_historical_networks()
        self.visualize_historical_networks(output_dir=output_dir, grid_size=grid_size)
        self.calculate_polarization_metrics()
        self.visualize_polarization_trends(output_dir=output_dir)
        
        print("Full historical analysis complete.")
        
        return self.metrics_df


if __name__ == "__main__":
    # Run historical analysis from 1965 to 2025
    analyzer = HistoricalPolarizationAnalyzer(start_year=1965, end_year=2025, step=4)
    analyzer.run_full_analysis()