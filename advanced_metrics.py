#!/usr/bin/env python3
"""
Advanced Congressional Polarization Metrics

This module calculates various advanced polarization metrics for congressional analysis,
including DW-NOMINATE-inspired scores, bipartisanship metrics, and clustering analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import os


class AdvancedPolarizationMetrics:
    def __init__(self, vote_matrix, member_data, chamber='house'):
        """
        Initialize with voting data and member information
        
        Parameters:
        -----------
        vote_matrix : pandas.DataFrame
            Member-by-vote matrix (rows are members, columns are votes)
        member_data : pandas.DataFrame
            DataFrame with member information
        chamber : str
            'house' or 'senate' (default: 'house')
        """
        self.vote_matrix = vote_matrix
        self.member_data = member_data
        self.chamber = chamber
        
        # Link the DataFrames
        if isinstance(self.member_data, pd.DataFrame) and not self.member_data.empty:
            self.member_data = self.member_data.set_index('member_id')
            
            # Make sure all members in vote_matrix have data in member_data
            missing_members = set(self.vote_matrix.index) - set(self.member_data.index)
            if missing_members:
                print(f"Warning: {len(missing_members)} members in vote matrix not found in member data.")
    
    def calculate_dw_nominate_inspired(self):
        """
        Calculate DW-NOMINATE-inspired scores using dimensionality reduction
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with member IDs and ideological positions
        """
        print(f"Calculating DW-NOMINATE-inspired scores for {self.chamber}...")
        
        # Use MDS (Multidimensional Scaling) to find a 2D representation
        # that preserves pairwise distances in voting behavior
        
        # First, convert vote matrix to distance matrix
        # 1. Fill missing values with the most common vote for that column
        vote_matrix_filled = self.vote_matrix.fillna(self.vote_matrix.mode().iloc[0])
        
        # 2. Calculate pairwise distances (Hamming distance works well for binary data)
        distances = pdist(vote_matrix_filled.values, metric='hamming')
        distance_matrix = squareform(distances)
        
        # 3. Apply MDS to get 2D coordinates
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distance_matrix)
        
        # Create a DataFrame with results
        dw_scores = pd.DataFrame(
            positions, 
            columns=['dimension1', 'dimension2'], 
            index=self.vote_matrix.index
        )
        
        # Add party information if available
        if isinstance(self.member_data, pd.DataFrame) and not self.member_data.empty:
            dw_scores['party'] = [
                self.member_data.loc[member_id, 'party'] if member_id in self.member_data.index else 'Unknown'
                for member_id in dw_scores.index
            ]
        
        return dw_scores
    
    def calculate_party_unity_scores(self):
        """
        Calculate how often members vote with their party's majority
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with party unity scores
        """
        print(f"Calculating party unity scores for {self.chamber}...")
        
        # Get party for each member
        if isinstance(self.member_data, pd.DataFrame) and not self.member_data.empty:
            member_parties = {
                idx: self.member_data.loc[idx, 'party'] if idx in self.member_data.index else 'Unknown'
                for idx in self.vote_matrix.index
            }
        else:
            raise ValueError("Member data required to calculate party unity scores")
        
        # For each vote, determine the majority position of each party
        party_majority = {}
        for vote_id in self.vote_matrix.columns:
            vote_col = self.vote_matrix[vote_id]
            
            # Group by party
            party_votes = {}
            for member_id, vote in vote_col.items():
                if pd.isna(vote):
                    continue
                    
                party = member_parties.get(member_id, 'Unknown')
                if party not in party_votes:
                    party_votes[party] = []
                    
                party_votes[party].append(vote)
            
            # Determine majority for each party
            vote_majority = {}
            for party, votes in party_votes.items():
                if not votes:
                    continue
                    
                # Count votes
                yes_count = votes.count(1)
                no_count = votes.count(0)
                
                # Determine majority
                if yes_count > no_count:
                    vote_majority[party] = 1
                elif no_count > yes_count:
                    vote_majority[party] = 0
                # If tied, no clear majority
            
            party_majority[vote_id] = vote_majority
        
        # Calculate unity score for each member
        unity_scores = {}
        for member_id in self.vote_matrix.index:
            party = member_parties.get(member_id, 'Unknown')
            if party == 'Unknown':
                continue
                
            votes_with_majority = 0
            total_votes = 0
            
            for vote_id in self.vote_matrix.columns:
                vote = self.vote_matrix.loc[member_id, vote_id]
                if pd.isna(vote):
                    continue
                    
                # Check if party had a majority for this vote
                if vote_id in party_majority and party in party_majority[vote_id]:
                    majority_vote = party_majority[vote_id][party]
                    
                    # Check if member voted with majority
                    if vote == majority_vote:
                        votes_with_majority += 1
                        
                    total_votes += 1
            
            # Calculate unity score
            if total_votes > 0:
                unity_scores[member_id] = (votes_with_majority / total_votes) * 100
        
        # Create DataFrame
        unity_df = pd.DataFrame({
            'member_id': list(unity_scores.keys()),
            'party_unity': list(unity_scores.values())
        })
        unity_df = unity_df.set_index('member_id')
        
        # Add party information
        unity_df['party'] = [member_parties.get(idx, 'Unknown') for idx in unity_df.index]
        
        return unity_df
    
    def calculate_bipartisan_scores(self):
        """
        Calculate how often members vote with the majority of the opposite party
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with bipartisan scores
        """
        print(f"Calculating bipartisanship scores for {self.chamber}...")
        
        # Get party for each member
        if isinstance(self.member_data, pd.DataFrame) and not self.member_data.empty:
            member_parties = {
                idx: self.member_data.loc[idx, 'party'] if idx in self.member_data.index else 'Unknown'
                for idx in self.vote_matrix.index
            }
        else:
            raise ValueError("Member data required to calculate bipartisan scores")
        
        # For each vote, determine the majority position of each party
        party_majority = {}
        for vote_id in self.vote_matrix.columns:
            vote_col = self.vote_matrix[vote_id]
            
            # Group by party
            party_votes = {}
            for member_id, vote in vote_col.items():
                if pd.isna(vote):
                    continue
                    
                party = member_parties.get(member_id, 'Unknown')
                if party not in party_votes:
                    party_votes[party] = []
                    
                party_votes[party].append(vote)
            
            # Determine majority for each party
            vote_majority = {}
            for party, votes in party_votes.items():
                if not votes:
                    continue
                    
                # Count votes
                yes_count = votes.count(1)
                no_count = votes.count(0)
                
                # Determine majority
                if yes_count > no_count:
                    vote_majority[party] = 1
                elif no_count > yes_count:
                    vote_majority[party] = 0
                # If tied, no clear majority
            
            party_majority[vote_id] = vote_majority
        
        # Calculate bipartisan score for each member
        bipartisan_scores = {}
        for member_id in self.vote_matrix.index:
            party = member_parties.get(member_id, 'Unknown')
            if party not in ['D', 'R']:  # Only calculate for major parties
                continue
                
            opposite_party = 'R' if party == 'D' else 'D'
            votes_with_opposite = 0
            total_votes = 0
            
            for vote_id in self.vote_matrix.columns:
                vote = self.vote_matrix.loc[member_id, vote_id]
                if pd.isna(vote):
                    continue
                    
                # Check if opposite party had a majority for this vote
                if vote_id in party_majority and opposite_party in party_majority[vote_id]:
                    opposite_majority = party_majority[vote_id][opposite_party]
                    
                    # Check if member voted with opposite party's majority
                    if vote == opposite_majority:
                        votes_with_opposite += 1
                        
                    total_votes += 1
            
            # Calculate bipartisan score
            if total_votes > 0:
                bipartisan_scores[member_id] = (votes_with_opposite / total_votes) * 100
        
        # Create DataFrame
        bipartisan_df = pd.DataFrame({
            'member_id': list(bipartisan_scores.keys()),
            'bipartisan_score': list(bipartisan_scores.values())
        })
        bipartisan_df = bipartisan_df.set_index('member_id')
        
        # Add party information
        bipartisan_df['party'] = [member_parties.get(idx, 'Unknown') for idx in bipartisan_df.index]
        
        return bipartisan_df
    
    def identify_voting_clusters(self, n_clusters=None):
        """
        Identify clusters of members with similar voting patterns
        
        Parameters:
        -----------
        n_clusters : int or None
            Number of clusters to identify (default: None, will be determined automatically)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster assignments and member information
        """
        print(f"Identifying voting clusters for {self.chamber}...")
        
        # Fill missing values (required for clustering)
        vote_matrix_filled = self.vote_matrix.fillna(self.vote_matrix.mode().iloc[0])
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            max_clusters = min(8, len(vote_matrix_filled) - 1)  # Don't try too many clusters
            silhouette_scores = []
            
            for k in range(2, max_clusters + 1):
                # Use KMeans for cluster analysis
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(vote_matrix_filled)
                
                # Calculate silhouette score to evaluate clustering quality
                if len(set(cluster_labels)) > 1:  # Need at least 2 different labels
                    score = silhouette_score(vote_matrix_filled, cluster_labels)
                    silhouette_scores.append((k, score))
            
            # Find best number of clusters
            if silhouette_scores:
                best_k, _ = max(silhouette_scores, key=lambda x: x[1])
                n_clusters = best_k
            else:
                n_clusters = 2  # Default to 2 if no clear optimal number
        
        # Perform clustering with optimal clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vote_matrix_filled)
        
        # Create DataFrame with results
        clusters_df = pd.DataFrame({
            'member_id': self.vote_matrix.index,
            'cluster': cluster_labels
        })
        clusters_df = clusters_df.set_index('member_id')
        
        # Add party information if available
        if isinstance(self.member_data, pd.DataFrame) and not self.member_data.empty:
            clusters_df['party'] = [
                self.member_data.loc[member_id, 'party'] if member_id in self.member_data.index else 'Unknown'
                for member_id in clusters_df.index
            ]
        
        return clusters_df
    
    def calculate_polarization_index(self, dw_scores=None):
        """
        Calculate polarization index based on ideological positions
        
        Parameters:
        -----------
        dw_scores : pandas.DataFrame or None
            DW-NOMINATE-inspired scores (default: None, will be calculated)
            
        Returns:
        --------
        dict
            Dictionary with polarization metrics
        """
        print(f"Calculating polarization index for {self.chamber}...")
        
        # Calculate DW-NOMINATE-inspired scores if not provided
        if dw_scores is None:
            dw_scores = self.calculate_dw_nominate_inspired()
        
        # Only proceed if we have party information
        if 'party' not in dw_scores.columns:
            raise ValueError("Party information required to calculate polarization index")
        
        # Get members by party (only consider Democrats and Republicans)
        dem_scores = dw_scores[dw_scores['party'] == 'D']
        rep_scores = dw_scores[dw_scores['party'] == 'R']
        
        # Calculate centroids for each party
        dem_centroid = dem_scores[['dimension1', 'dimension2']].mean()
        rep_centroid = rep_scores[['dimension1', 'dimension2']].mean()
        
        # Calculate distance between centroids
        centroid_distance = np.sqrt(
            (dem_centroid['dimension1'] - rep_centroid['dimension1'])**2 +
            (dem_centroid['dimension2'] - rep_centroid['dimension2'])**2
        )
        
        # Calculate dispersion within each party (average distance to centroid)
        dem_dispersion = np.mean([
            np.sqrt(
                (row['dimension1'] - dem_centroid['dimension1'])**2 +
                (row['dimension2'] - dem_centroid['dimension2'])**2
            )
            for _, row in dem_scores.iterrows()
        ])
        
        rep_dispersion = np.mean([
            np.sqrt(
                (row['dimension1'] - rep_centroid['dimension1'])**2 +
                (row['dimension2'] - rep_centroid['dimension2'])**2
            )
            for _, row in rep_scores.iterrows()
        ])
        
        # Calculate average within-party dispersion
        average_dispersion = (dem_dispersion + rep_dispersion) / 2
        
        # Calculate polarization index (distance between centroids / average within-party dispersion)
        polarization_index = centroid_distance / average_dispersion if average_dispersion > 0 else float('inf')
        
        # Calculate overlap between parties (proportion of members whose ideological position is closer to the
        # opposite party's centroid than to their own party's centroid)
        dem_closer_to_rep = sum(
            np.sqrt(
                (row['dimension1'] - rep_centroid['dimension1'])**2 +
                (row['dimension2'] - rep_centroid['dimension2'])**2
            ) < np.sqrt(
                (row['dimension1'] - dem_centroid['dimension1'])**2 +
                (row['dimension2'] - dem_centroid['dimension2'])**2
            )
            for _, row in dem_scores.iterrows()
        )
        
        rep_closer_to_dem = sum(
            np.sqrt(
                (row['dimension1'] - dem_centroid['dimension1'])**2 +
                (row['dimension2'] - dem_centroid['dimension2'])**2
            ) < np.sqrt(
                (row['dimension1'] - rep_centroid['dimension1'])**2 +
                (row['dimension2'] - rep_centroid['dimension2'])**2
            )
            for _, row in rep_scores.iterrows()
        )
        
        total_members = len(dem_scores) + len(rep_scores)
        overlap_proportion = (dem_closer_to_rep + rep_closer_to_dem) / total_members if total_members > 0 else 0
        
        # Calculate party separation (minimum distance between any Democrat and Republican)
        min_distance = float('inf')
        for _, dem_row in dem_scores.iterrows():
            for _, rep_row in rep_scores.iterrows():
                distance = np.sqrt(
                    (dem_row['dimension1'] - rep_row['dimension1'])**2 +
                    (dem_row['dimension2'] - rep_row['dimension2'])**2
                )
                min_distance = min(min_distance, distance)
        
        # Return metrics
        return {
            'centroid_distance': centroid_distance,
            'dem_dispersion': dem_dispersion,
            'rep_dispersion': rep_dispersion,
            'average_dispersion': average_dispersion,
            'polarization_index': polarization_index,
            'overlap_proportion': overlap_proportion,
            'min_party_distance': min_distance
        }
    
    def calculate_vote_polarization(self):
        """
        Calculate polarization metrics for each vote
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with polarization metrics for each vote
        """
        print(f"Calculating vote polarization for {self.chamber}...")
        
        # Get party for each member
        if isinstance(self.member_data, pd.DataFrame) and not self.member_data.empty:
            member_parties = {
                idx: self.member_data.loc[idx, 'party'] if idx in self.member_data.index else 'Unknown'
                for idx in self.vote_matrix.index
            }
        else:
            raise ValueError("Member data required to calculate vote polarization")
        
        vote_metrics = []
        
        for vote_id in self.vote_matrix.columns:
            vote_col = self.vote_matrix[vote_id]
            
            # Count votes by party
            dem_votes = {'yes': 0, 'no': 0}
            rep_votes = {'yes': 0, 'no': 0}
            
            for member_id, vote in vote_col.items():
                if pd.isna(vote):
                    continue
                    
                party = member_parties.get(member_id, 'Unknown')
                vote_str = 'yes' if vote == 1 else 'no'
                
                if party == 'D':
                    dem_votes[vote_str] += 1
                elif party == 'R':
                    rep_votes[vote_str] += 1
            
            # Calculate party unity on this vote
            dem_total = dem_votes['yes'] + dem_votes['no']
            rep_total = rep_votes['yes'] + rep_votes['no']
            
            dem_unity = max(dem_votes['yes'], dem_votes['no']) / dem_total if dem_total > 0 else 0
            rep_unity = max(rep_votes['yes'], rep_votes['no']) / rep_total if rep_total > 0 else 0
            
            # Determine if parties voted in opposite directions
            dem_majority = 'yes' if dem_votes['yes'] > dem_votes['no'] else 'no'
            rep_majority = 'yes' if rep_votes['yes'] > rep_votes['no'] else 'no'
            
            party_opposition = dem_majority != rep_majority
            
            # Calculate overall polarization for this vote
            total_votes = dem_total + rep_total
            
            # Polarization is high when parties vote in opposite directions with high unity
            if party_opposition:
                vote_polarization = (dem_unity * dem_total + rep_unity * rep_total) / total_votes if total_votes > 0 else 0
            else:
                vote_polarization = 0
            
            vote_metrics.append({
                'vote_id': vote_id,
                'dem_yes': dem_votes['yes'],
                'dem_no': dem_votes['no'],
                'rep_yes': rep_votes['yes'],
                'rep_no': rep_votes['no'],
                'dem_unity': dem_unity,
                'rep_unity': rep_unity,
                'party_opposition': party_opposition,
                'vote_polarization': vote_polarization
            })
        
        return pd.DataFrame(vote_metrics)
    
    def visualize_ideological_space(self, dw_scores=None, output_dir=None, filename=None):
        """
        Visualize members in ideological space
        
        Parameters:
        -----------
        dw_scores : pandas.DataFrame or None
            DW-NOMINATE-inspired scores (default: None, will be calculated)
        output_dir : str or None
            Directory to save visualization (default: None, will display instead)
        filename : str or None
            Filename for saved visualization (default: None)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Calculate DW-NOMINATE-inspired scores if not provided
        if dw_scores is None:
            dw_scores = self.calculate_dw_nominate_inspired()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define colors for parties
        party_colors = {
            'D': 'blue',
            'R': 'red',
            'I': 'green',
            'Unknown': 'gray'
        }
        
        # Plot each member
        for party in dw_scores['party'].unique():
            party_data = dw_scores[dw_scores['party'] == party]
            ax.scatter(
                party_data['dimension1'],
                party_data['dimension2'],
                c=party_colors.get(party, 'gray'),
                label=party,
                alpha=0.7,
                s=80
            )
        
        # Calculate and plot centroids for major parties
        for party in ['D', 'R']:
            if party in dw_scores['party'].values:
                party_data = dw_scores[dw_scores['party'] == party]
                centroid = party_data[['dimension1', 'dimension2']].mean()
                ax.scatter(
                    centroid['dimension1'],
                    centroid['dimension2'],
                    c=party_colors.get(party, 'gray'),
                    marker='X',
                    s=200,
                    edgecolors='black',
                    linewidths=2,
                    label=f"{party} Centroid"
                )
        
        # Add title and labels
        ax.set_title(f'Ideological Space - {self.chamber.capitalize()}', fontsize=16)
        ax.set_xlabel('First Dimension', fontsize=14)
        ax.set_ylabel('Second Dimension', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        
        # Save or display
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                filename = f"{self.chamber}_ideological_space.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
        
        return fig
    
    def run_all_analyses(self, output_dir=None):
        """
        Run all polarization analyses
        
        Parameters:
        -----------
        output_dir : str or None
            Directory to save visualizations (default: None)
            
        Returns:
        --------
        dict
            Dictionary with all analysis results
        """
        print(f"Running all polarization analyses for {self.chamber}...")
        
        # Create directories if needed
        if output_dir is not None:
            chamber_dir = os.path.join(output_dir, self.chamber)
            os.makedirs(chamber_dir, exist_ok=True)
        else:
            chamber_dir = None
        
        # Run all analyses
        dw_scores = self.calculate_dw_nominate_inspired()
        unity_scores = self.calculate_party_unity_scores()
        bipartisan_scores = self.calculate_bipartisan_scores()
        clusters = self.identify_voting_clusters()
        polarization_index = self.calculate_polarization_index(dw_scores)
        vote_polarization = self.calculate_vote_polarization()
        
        # Create visualizations
        if chamber_dir is not None:
            self.visualize_ideological_space(dw_scores, chamber_dir, 'ideological_space.png')
            
            # Visualize party unity
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='party', y='party_unity', data=unity_scores)
            plt.title(f'Party Unity Scores - {self.chamber.capitalize()}', fontsize=16)
            plt.xlabel('Party', fontsize=14)
            plt.ylabel('Party Unity Score (%)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(chamber_dir, 'party_unity.png'), dpi=300)
            plt.close()
            
            # Visualize bipartisan scores
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='party', y='bipartisan_score', data=bipartisan_scores)
            plt.title(f'Bipartisanship Scores - {self.chamber.capitalize()}', fontsize=16)
            plt.xlabel('Party', fontsize=14)
            plt.ylabel('Bipartisanship Score (%)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(chamber_dir, 'bipartisanship.png'), dpi=300)
            plt.close()
            
            # Visualize clusters
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Use DW scores for cluster visualization
            merged_data = pd.merge(
                clusters, 
                dw_scores, 
                left_index=True, 
                right_index=True,
                suffixes=('_cluster', '')
            )
            
            # Plot clusters
            for cluster in merged_data['cluster'].unique():
                cluster_data = merged_data[merged_data['cluster'] == cluster]
                ax.scatter(
                    cluster_data['dimension1'],
                    cluster_data['dimension2'],
                    label=f'Cluster {cluster}',
                    alpha=0.7,
                    s=80
                )
            
            ax.set_title(f'Voting Clusters - {self.chamber.capitalize()}', fontsize=16)
            ax.set_xlabel('First Dimension', fontsize=14)
            ax.set_ylabel('Second Dimension', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            plt.savefig(os.path.join(chamber_dir, 'voting_clusters.png'), dpi=300)
            plt.close()
            
            # Visualize vote polarization
            plt.figure(figsize=(12, 6))
            sns.histplot(vote_polarization['vote_polarization'], bins=20)
            plt.axvline(vote_polarization['vote_polarization'].mean(), color='red', linestyle='--',
                        label=f'Mean: {vote_polarization["vote_polarization"].mean():.2f}')
            plt.title(f'Vote Polarization Distribution - {self.chamber.capitalize()}', fontsize=16)
            plt.xlabel('Polarization Score', fontsize=14)
            plt.ylabel('Number of Votes', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig(os.path.join(chamber_dir, 'vote_polarization.png'), dpi=300)
            plt.close()
        
        # Return all results
        return {
            'dw_scores': dw_scores,
            'unity_scores': unity_scores,
            'bipartisan_scores': bipartisan_scores,
            'clusters': clusters,
            'polarization_index': polarization_index,
            'vote_polarization': vote_polarization
        }


if __name__ == "__main__":
    # Example usage
    from congressional_polarization import CongressionalPolarizationAnalyzer
    
    # Create analyzer and run initial analysis
    analyzer = CongressionalPolarizationAnalyzer(congress_number=119)
    analyzer.fetch_congressional_data()
    analyzer.create_vote_matrices()
    
    # Run advanced metrics for House
    house_metrics = AdvancedPolarizationMetrics(
        analyzer.house_vote_matrix,
        analyzer.house_members_df,
        chamber='house'
    )
    house_results = house_metrics.run_all_analyses(output_dir='results/advanced_metrics')
    
    # Run advanced metrics for Senate
    senate_metrics = AdvancedPolarizationMetrics(
        analyzer.senate_vote_matrix,
        analyzer.senate_members_df,
        chamber='senate'
    )
    senate_results = senate_metrics.run_all_analyses(output_dir='results/advanced_metrics')
    
    # Print polarization indices
    print("\nPolarization Indices:")
    print(f"House: {house_results['polarization_index']['polarization_index']:.2f}")
    print(f"Senate: {senate_results['polarization_index']['polarization_index']:.2f}")