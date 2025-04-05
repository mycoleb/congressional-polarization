#!/usr/bin/env python3
"""
Congressional Polarization Analysis

This module analyzes voting patterns and polarization in the current US Congress,
including network analysis and visualization of ideological space.
"""

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from io import StringIO
import os
from tqdm import tqdm


class CongressionalPolarizationAnalyzer:
    def __init__(self, congress_number=119):
        """
        Initialize the analyzer with the congress number to analyze
        
        Parameters:
        -----------
        congress_number : int
            The number of the congress to analyze (default: 119 for the 119th Congress)
        """
        self.congress_number = congress_number
        self.house_votes_df = None
        self.senate_votes_df = None
        self.house_members_df = None
        self.senate_members_df = None
        self.house_vote_matrix = None
        self.senate_vote_matrix = None
        self.house_similarity_matrix = None
        self.senate_similarity_matrix = None
        self.house_graph = None
        self.senate_graph = None
    
    def fetch_congressional_data(self):
        """
        Fetch voting and member data for the specified Congress from ProPublica API
        Note: This requires an API key from ProPublica (https://www.propublica.org/datastore/api/propublica-congress-api)
        """
        # For the purpose of this demo, we'll simulate data
        # In a real implementation, you would use:
        # 
        # headers = {'X-API-Key': 'YOUR_API_KEY'}
        # url = f'https://api.propublica.org/congress/v1/{self.congress_number}/house/members.json'
        # response = requests.get(url, headers=headers)
        # data = response.json()
        
        print(f"Simulating data fetching for the {self.congress_number}th Congress...")
        
        # Simulate House members data (in reality, fetch from API)
        house_members = []
        for i in range(435):  # 435 House representatives
            party = 'R' if i < 218 else 'D'  # Simulating a narrow Republican majority
            
            # Add some variation to ideology scores
            if party == 'R':
                ideology = np.random.normal(0.7, 0.15)  # Right-leaning
            else:
                ideology = np.random.normal(-0.7, 0.15)  # Left-leaning
                
            house_members.append({
                'id': f'H{i+1}',
                'name': f'Representative {i+1}',
                'party': party,
                'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']),
                'ideology_score': ideology
            })
        
        # Simulate Senate members data (in reality, fetch from API)
        senate_members = []
        for i in range(100):  # 100 Senators
            party = 'R' if i < 49 else 'D' if i < 98 else 'I'  # Simulating Democratic majority with 2 Independents
            
            # Add some variation to ideology scores
            if party == 'R':
                ideology = np.random.normal(0.7, 0.15)  # Right-leaning
            elif party == 'D':
                ideology = np.random.normal(-0.7, 0.15)  # Left-leaning
            else:
                ideology = np.random.normal(-0.3, 0.1)  # Independent, slightly left-leaning
                
            senate_members.append({
                'id': f'S{i+1}',
                'name': f'Senator {i+1}',
                'party': party,
                'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']),
                'ideology_score': ideology
            })
        
        self.house_members_df = pd.DataFrame(house_members)
        self.senate_members_df = pd.DataFrame(senate_members)
        
        # Simulate voting data
        # Let's create 100 votes for demonstration purposes
        num_votes = 100
        
        # For the House
        house_votes = []
        for vote_id in range(num_votes):
            # Create a vote with increasing polarization
            # Higher vote_id means more partisan voting
            polarization_factor = 0.5 + (vote_id / num_votes) * 0.5
            
            for member_id in self.house_members_df['id']:
                member_idx = int(member_id[1:]) - 1
                member_party = self.house_members_df.loc[member_idx, 'party']
                member_ideology = self.house_members_df.loc[member_idx, 'ideology_score']
                
                # Determine probability of voting 'yes' based on party and ideology
                if vote_id % 2 == 0:  # Republican-sponsored bill
                    if member_party == 'R':
                        prob_yes = 0.9 - (1 - polarization_factor) * 0.4
                    else:
                        prob_yes = 0.1 + (1 - polarization_factor) * 0.4
                else:  # Democrat-sponsored bill
                    if member_party == 'D':
                        prob_yes = 0.9 - (1 - polarization_factor) * 0.4
                    else:
                        prob_yes = 0.1 + (1 - polarization_factor) * 0.4
                
                # Adjust based on ideology score
                prob_yes += member_ideology * 0.1 * (-1 if vote_id % 2 == 1 else 1)
                prob_yes = max(0.05, min(0.95, prob_yes))
                
                vote = np.random.choice(['Yes', 'No'], p=[prob_yes, 1-prob_yes])
                
                house_votes.append({
                    'vote_id': f'HV{vote_id+1}',
                    'member_id': member_id,
                    'vote': vote,
                    'party': member_party
                })
        
        # For the Senate
        senate_votes = []
        for vote_id in range(num_votes):
            # Create a vote with increasing polarization
            polarization_factor = 0.5 + (vote_id / num_votes) * 0.5
            
            for member_id in self.senate_members_df['id']:
                member_idx = int(member_id[1:]) - 1
                member_party = self.senate_members_df.loc[member_idx, 'party']
                member_ideology = self.senate_members_df.loc[member_idx, 'ideology_score']
                
                # Determine probability of voting 'yes' based on party and ideology
                if vote_id % 2 == 0:  # Republican-sponsored bill
                    if member_party == 'R':
                        prob_yes = 0.9 - (1 - polarization_factor) * 0.4
                    elif member_party == 'D':
                        prob_yes = 0.1 + (1 - polarization_factor) * 0.4
                    else:  # Independents
                        prob_yes = 0.3 + (1 - polarization_factor) * 0.4
                else:  # Democrat-sponsored bill
                    if member_party == 'D':
                        prob_yes = 0.9 - (1 - polarization_factor) * 0.4
                    elif member_party == 'R':
                        prob_yes = 0.1 + (1 - polarization_factor) * 0.4
                    else:  # Independents
                        prob_yes = 0.7 - (1 - polarization_factor) * 0.4
                
                # Adjust based on ideology score
                prob_yes += member_ideology * 0.1 * (-1 if vote_id % 2 == 1 else 1)
                prob_yes = max(0.05, min(0.95, prob_yes))
                
                vote = np.random.choice(['Yes', 'No'], p=[prob_yes, 1-prob_yes])
                
                senate_votes.append({
                    'vote_id': f'SV{vote_id+1}',
                    'member_id': member_id,
                    'vote': vote,
                    'party': member_party
                })
        
        self.house_votes_df = pd.DataFrame(house_votes)
        self.senate_votes_df = pd.DataFrame(senate_votes)
        
        print("Data fetching simulation complete.")
        
    def create_vote_matrices(self):
        """
        Transform voting data into member-by-vote matrices
        """
        if self.house_votes_df is None or self.senate_votes_df is None:
            raise ValueError("Vote data not loaded. Run fetch_congressional_data() first.")
        
        print("Creating vote matrices...")
        
        # Create House vote matrix
        house_vote_matrix = self.house_votes_df.pivot(
            index='member_id', 
            columns='vote_id', 
            values='vote'
        )
        
        # Convert votes to binary (1 for Yes, 0 for No)
        house_vote_matrix = (house_vote_matrix == 'Yes').astype(int)
        
        # Create Senate vote matrix
        senate_vote_matrix = self.senate_votes_df.pivot(
            index='member_id', 
            columns='vote_id', 
            values='vote'
        )
        
        # Convert votes to binary (1 for Yes, 0 for No)
        senate_vote_matrix = (senate_vote_matrix == 'Yes').astype(int)
        
        self.house_vote_matrix = house_vote_matrix
        self.senate_vote_matrix = senate_vote_matrix
        
        print("Vote matrices created.")
    
    def calculate_similarity(self):
        """
        Calculate similarity matrices based on voting patterns
        """
        if self.house_vote_matrix is None or self.senate_vote_matrix is None:
            raise ValueError("Vote matrices not created. Run create_vote_matrices() first.")
        
        print("Calculating similarity matrices...")
        
        # For House
        house_similarity = np.zeros((len(self.house_vote_matrix), len(self.house_vote_matrix)))
        
        for i, member1 in enumerate(self.house_vote_matrix.index):
            for j, member2 in enumerate(self.house_vote_matrix.index):
                if i <= j:  # We only need to calculate half the matrix due to symmetry
                    agreement = (self.house_vote_matrix.loc[member1] == self.house_vote_matrix.loc[member2]).mean()
                    house_similarity[i, j] = house_similarity[j, i] = agreement
        
        # For Senate
        senate_similarity = np.zeros((len(self.senate_vote_matrix), len(self.senate_vote_matrix)))
        
        for i, member1 in enumerate(self.senate_vote_matrix.index):
            for j, member2 in enumerate(self.senate_vote_matrix.index):
                if i <= j:  # We only need to calculate half the matrix due to symmetry
                    agreement = (self.senate_vote_matrix.loc[member1] == self.senate_vote_matrix.loc[member2]).mean()
                    senate_similarity[i, j] = senate_similarity[j, i] = agreement
        
        self.house_similarity_matrix = pd.DataFrame(
            house_similarity, 
            index=self.house_vote_matrix.index, 
            columns=self.house_vote_matrix.index
        )
        
        self.senate_similarity_matrix = pd.DataFrame(
            senate_similarity, 
            index=self.senate_vote_matrix.index, 
            columns=self.senate_vote_matrix.index
        )
        
        print("Similarity matrices calculated.")
    
    def create_networks(self, threshold=0.7):
        """
        Create network graphs based on similarity matrices
        
        Parameters:
        -----------
        threshold : float
            Similarity threshold for creating edges (default: 0.7)
        """
        if self.house_similarity_matrix is None or self.senate_similarity_matrix is None:
            raise ValueError("Similarity matrices not calculated. Run calculate_similarity() first.")
        
        print(f"Creating network graphs with threshold {threshold}...")
        
        # Create House network
        house_graph = nx.Graph()
        
        # Add nodes with attributes
        for member_id in self.house_similarity_matrix.index:
            member_idx = int(member_id[1:]) - 1
            party = self.house_members_df.loc[member_idx, 'party']
            house_graph.add_node(member_id, party=party)
        
        # Add edges based on similarity threshold
        for i, member1 in enumerate(self.house_similarity_matrix.index):
            for j, member2 in enumerate(self.house_similarity_matrix.index):
                if i < j and self.house_similarity_matrix.loc[member1, member2] >= threshold:
                    weight = self.house_similarity_matrix.loc[member1, member2]
                    house_graph.add_edge(member1, member2, weight=weight)
        
        # Create Senate network
        senate_graph = nx.Graph()
        
        # Add nodes with attributes
        for member_id in self.senate_similarity_matrix.index:
            member_idx = int(member_id[1:]) - 1
            party = self.senate_members_df.loc[member_idx, 'party']
            senate_graph.add_node(member_id, party=party)
        
        # Add edges based on similarity threshold
        for i, member1 in enumerate(self.senate_similarity_matrix.index):
            for j, member2 in enumerate(self.senate_similarity_matrix.index):
                if i < j and self.senate_similarity_matrix.loc[member1, member2] >= threshold:
                    weight = self.senate_similarity_matrix.loc[member1, member2]
                    senate_graph.add_edge(member1, member2, weight=weight)
        
        self.house_graph = house_graph
        self.senate_graph = senate_graph
        
        print("Network graphs created.")
    
    def get_party_color(self, party):
        """Map party to color"""
        if party == 'D':
            return 'blue'
        elif party == 'R':
            return 'red'
        else:
            return 'green'  # Independent
    
    def visualize_networks(self, output_dir='visualizations'):
        """
        Visualize the network graphs
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations (default: 'visualizations')
        """
        if self.house_graph is None or self.senate_graph is None:
            raise ValueError("Network graphs not created. Run create_networks() first.")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Visualizing networks...")
        
        # Visualize House network
        plt.figure(figsize=(12, 10))
        
        # Use force-directed layout
        pos = nx.spring_layout(self.house_graph, k=0.2, iterations=50)
        
        # Get node colors based on party
        colors = [self.get_party_color(self.house_graph.nodes[node]['party']) for node in self.house_graph.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.house_graph, 
            pos, 
            node_size=50, 
            node_color=colors, 
            alpha=0.8
        )
        
        # Draw edges with low alpha for visibility
        nx.draw_networkx_edges(
            self.house_graph, 
            pos, 
            width=0.5, 
            alpha=0.2
        )
        
        plt.title(f'House of Representatives - {self.congress_number}th Congress')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/house_network.png', dpi=300)
        plt.close()
        
        # Visualize Senate network
        plt.figure(figsize=(12, 10))
        
        # Use force-directed layout
        pos = nx.spring_layout(self.senate_graph, k=0.3, iterations=50)
        
        # Get node colors based on party
        colors = [self.get_party_color(self.senate_graph.nodes[node]['party']) for node in self.senate_graph.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.senate_graph, 
            pos, 
            node_size=100, 
            node_color=colors, 
            alpha=0.8
        )
        
        # Draw edges with low alpha for visibility
        nx.draw_networkx_edges(
            self.senate_graph, 
            pos, 
            width=0.5, 
            alpha=0.2
        )
        
        plt.title(f'Senate - {self.congress_number}th Congress')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/senate_network.png', dpi=300)
        plt.close()
        
        print(f"Network visualizations saved to {output_dir}/")
    
    def perform_dimensional_reduction(self):
        """
        Perform dimensional reduction on voting data using PCA
        """
        if self.house_vote_matrix is None or self.senate_vote_matrix is None:
            raise ValueError("Vote matrices not created. Run create_vote_matrices() first.")
        
        print("Performing dimensional reduction...")
        
        # For House
        house_pca = PCA(n_components=2)
        house_coords = house_pca.fit_transform(self.house_vote_matrix)
        
        house_positions = pd.DataFrame(
            house_coords, 
            columns=['x', 'y'], 
            index=self.house_vote_matrix.index
        )
        
        # Add party information
        house_positions['party'] = [
            self.house_members_df.loc[int(member_id[1:]) - 1, 'party'] 
            for member_id in house_positions.index
        ]
        
        # For Senate
        senate_pca = PCA(n_components=2)
        senate_coords = senate_pca.fit_transform(self.senate_vote_matrix)
        
        senate_positions = pd.DataFrame(
            senate_coords, 
            columns=['x', 'y'], 
            index=self.senate_vote_matrix.index
        )
        
        # Add party information
        senate_positions['party'] = [
            self.senate_members_df.loc[int(member_id[1:]) - 1, 'party'] 
            for member_id in senate_positions.index
        ]
        
        # Store results
        self.house_positions = house_positions
        self.senate_positions = senate_positions
        
        # Calculate explained variance
        self.house_explained_variance = house_pca.explained_variance_ratio_
        self.senate_explained_variance = senate_pca.explained_variance_ratio_
        
        print("Dimensional reduction complete.")
        print(f"House explained variance: {self.house_explained_variance[0]:.2f}, {self.house_explained_variance[1]:.2f}")
        print(f"Senate explained variance: {self.senate_explained_variance[0]:.2f}, {self.senate_explained_variance[1]:.2f}")
    
    def visualize_ideological_space(self, output_dir='visualizations'):
        """
        Visualize the ideological space based on dimensional reduction
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations (default: 'visualizations')
        """
        if not hasattr(self, 'house_positions') or not hasattr(self, 'senate_positions'):
            raise ValueError("Dimensional reduction not performed. Run perform_dimensional_reduction() first.")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Visualizing ideological space...")
        
        # For House
        plt.figure(figsize=(10, 8))
        
        # Plot points colored by party
        for party, color in [('D', 'blue'), ('R', 'red'), ('I', 'green')]:
            party_data = self.house_positions[self.house_positions['party'] == party]
            plt.scatter(
                party_data['x'], 
                party_data['y'], 
                c=color, 
                alpha=0.7, 
                s=50, 
                label=f"{party}"
            )
        
        plt.title(f'Ideological Space - House of Representatives ({self.congress_number}th Congress)')
        plt.xlabel(f'First dimension ({self.house_explained_variance[0]:.2%} variance)')
        plt.ylabel(f'Second dimension ({self.house_explained_variance[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{output_dir}/house_ideological_space.png', dpi=300)
        plt.close()
        
        # For Senate
        plt.figure(figsize=(10, 8))
        
        # Plot points colored by party
        for party, color in [('D', 'blue'), ('R', 'red'), ('I', 'green')]:
            party_data = self.senate_positions[self.senate_positions['party'] == party]
            plt.scatter(
                party_data['x'], 
                party_data['y'], 
                c=color, 
                alpha=0.7, 
                s=70, 
                label=f"{party}"
            )
        
        plt.title(f'Ideological Space - Senate ({self.congress_number}th Congress)')
        plt.xlabel(f'First dimension ({self.senate_explained_variance[0]:.2%} variance)')
        plt.ylabel(f'Second dimension ({self.senate_explained_variance[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{output_dir}/senate_ideological_space.png', dpi=300)
        plt.close()
        
        print(f"Ideological space visualizations saved to {output_dir}/")
    
    def calculate_polarization_metrics(self):
        """
        Calculate various polarization metrics
        """
        if not hasattr(self, 'house_positions') or not hasattr(self, 'senate_positions'):
            raise ValueError("Dimensional reduction not performed. Run perform_dimensional_reduction() first.")
        
        print("Calculating polarization metrics...")
        
        # For House
        # 1. Calculate distance between party centroids
        house_dem_centroid = self.house_positions[self.house_positions['party'] == 'D'][['x', 'y']].mean()
        house_rep_centroid = self.house_positions[self.house_positions['party'] == 'R'][['x', 'y']].mean()
        
        house_centroid_distance = np.sqrt(
            (house_dem_centroid['x'] - house_rep_centroid['x'])**2 + 
            (house_dem_centroid['y'] - house_rep_centroid['y'])**2
        )
        
        # 2. Calculate within-party cohesion (average distance to own party centroid)
        house_dem_cohesion = np.mean([
            np.sqrt((row['x'] - house_dem_centroid['x'])**2 + (row['y'] - house_dem_centroid['y'])**2)
            for _, row in self.house_positions[self.house_positions['party'] == 'D'].iterrows()
        ])
        
        house_rep_cohesion = np.mean([
            np.sqrt((row['x'] - house_rep_centroid['x'])**2 + (row['y'] - house_rep_centroid['y'])**2)
            for _, row in self.house_positions[self.house_positions['party'] == 'R'].iterrows()
        ])
        
        # 3. Calculate polarization index (distance between centroids / average within-party distance)
        house_polarization_index = house_centroid_distance / ((house_dem_cohesion + house_rep_cohesion) / 2)
        
        # For Senate
        # 1. Calculate distance between party centroids
        senate_dem_centroid = self.senate_positions[self.senate_positions['party'] == 'D'][['x', 'y']].mean()
        senate_rep_centroid = self.senate_positions[self.senate_positions['party'] == 'R'][['x', 'y']].mean()
        
        senate_centroid_distance = np.sqrt(
            (senate_dem_centroid['x'] - senate_rep_centroid['x'])**2 + 
            (senate_dem_centroid['y'] - senate_rep_centroid['y'])**2
        )
        
        # 2. Calculate within-party cohesion (average distance to own party centroid)
        senate_dem_cohesion = np.mean([
            np.sqrt((row['x'] - senate_dem_centroid['x'])**2 + (row['y'] - senate_dem_centroid['y'])**2)
            for _, row in self.senate_positions[self.senate_positions['party'] == 'D'].iterrows()
        ])
        
        senate_rep_cohesion = np.mean([
            np.sqrt((row['x'] - senate_rep_centroid['x'])**2 + (row['y'] - senate_rep_centroid['y'])**2)
            for _, row in self.senate_positions[self.senate_positions['party'] == 'R'].iterrows()
        ])
        
        # 3. Calculate polarization index (distance between centroids / average within-party distance)
        senate_polarization_index = senate_centroid_distance / ((senate_dem_cohesion + senate_rep_cohesion) / 2)
        
        # Store metrics
        self.polarization_metrics = {
            'house': {
                'dem_rep_distance': house_centroid_distance,
                'dem_cohesion': house_dem_cohesion,
                'rep_cohesion': house_rep_cohesion,
                'polarization_index': house_polarization_index
            },
            'senate': {
                'dem_rep_distance': senate_centroid_distance,
                'dem_cohesion': senate_dem_cohesion,
                'rep_cohesion': senate_rep_cohesion,
                'polarization_index': senate_polarization_index
            }
        }
        
        print("Polarization metrics calculated.")
        print(f"House polarization index: {house_polarization_index:.2f}")
        print(f"Senate polarization index: {senate_polarization_index:.2f}")
    
    def run_full_analysis(self, threshold=0.7, output_dir='visualizations'):
        """
        Run the full analysis pipeline
        
        Parameters:
        -----------
        threshold : float
            Similarity threshold for creating network edges (default: 0.7)
        output_dir : str
            Directory to save visualizations (default: 'visualizations')
        """
        print(f"Running full analysis for the {self.congress_number}th Congress...")
        
        self.fetch_congressional_data()
        self.create_vote_matrices()
        self.calculate_similarity()
        self.create_networks(threshold=threshold)
        self.visualize_networks(output_dir=output_dir)
        self.perform_dimensional_reduction()
        self.visualize_ideological_space(output_dir=output_dir)
        self.calculate_polarization_metrics()
        
        print("Full analysis complete.")
        
        return {
            'house_polarization': self.polarization_metrics['house']['polarization_index'],
            'senate_polarization': self.polarization_metrics['senate']['polarization_index']
        }


if __name__ == "__main__":
    # Analyze just the 119th Congress
    analyzer = CongressionalPolarizationAnalyzer(congress_number=119)
    analyzer.run_full_analysis()