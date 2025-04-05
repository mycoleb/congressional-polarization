#!/usr/bin/env python3
"""
Congressional Data Ingestion

This script provides functions to load real congressional data fetched by data_fetcher.py
and prepare it for analysis by the main analysis scripts.
"""

import os
import pandas as pd
import numpy as np


def load_real_congressional_data(data_dir):
    """
    Load congressional data from CSV files
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    tuple
        (house_members_df, senate_members_df, house_votes_df, senate_votes_df)
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")
    
    # Load member data
    house_members_path = os.path.join(data_dir, 'house_members.csv')
    senate_members_path = os.path.join(data_dir, 'senate_members.csv')
    
    if not os.path.exists(house_members_path) or not os.path.exists(senate_members_path):
        raise FileNotFoundError(f"Member data files not found in '{data_dir}'")
    
    house_members_df = pd.read_csv(house_members_path)
    senate_members_df = pd.read_csv(senate_members_path)
    
    # Load vote data
    house_votes_path = os.path.join(data_dir, 'house_votes.csv')
    senate_votes_path = os.path.join(data_dir, 'senate_votes.csv')
    
    if not os.path.exists(house_votes_path) or not os.path.exists(senate_votes_path):
        raise FileNotFoundError(f"Vote data files not found in '{data_dir}'")
    
    house_votes_df = pd.read_csv(house_votes_path)
    senate_votes_df = pd.read_csv(senate_votes_path)
    
    # Add 'id' prefix to member_id in votes dataframes for compatibility
    house_votes_df['member_id'] = 'H' + house_votes_df['member_id'].astype(str)
    senate_votes_df['member_id'] = 'S' + senate_votes_df['member_id'].astype(str)
    
    # Add member_id column to members dataframes
    house_members_df['member_id'] = 'H' + house_members_df['id'].astype(str)
    senate_members_df['member_id'] = 'S' + senate_members_df['id'].astype(str)
    
    # Map vote positions to 'Yes'/'No' for analysis
    house_votes_df['vote'] = house_votes_df['vote_position'].map({
        'Yes': 'Yes',
        'No': 'No',
        'Present': 'No',
        'Not Voting': np.nan
    })
    
    senate_votes_df['vote'] = senate_votes_df['vote_position'].map({
        'Yes': 'Yes',
        'No': 'No',
        'Present': 'No',
        'Not Voting': np.nan
    })
    
    # Drop rows with NaN votes
    house_votes_df = house_votes_df.dropna(subset=['vote'])
    senate_votes_df = senate_votes_df.dropna(subset=['vote'])
    
    # Create vote_id prefixes for uniqueness
    house_votes_df['vote_id'] = 'HV' + house_votes_df['vote_id'].astype(str)
    senate_votes_df['vote_id'] = 'SV' + senate_votes_df['vote_id'].astype(str)
    
    # Add ideology scores based on party and votes with party percentage
    # This is a simplification; real ideology scores would require more complex analysis
    house_members_df['ideology_score'] = house_members_df.apply(
        lambda row: (row['votes_with_party_pct'] / 100) * (1 if row['party'] == 'R' else -1),
        axis=1
    )
    
    senate_members_df['ideology_score'] = senate_members_df.apply(
        lambda row: (row['votes_with_party_pct'] / 100) * (1 if row['party'] == 'R' else -1),
        axis=1
    )
    
    return house_members_df, senate_members_df, house_votes_df, senate_votes_df


def prepare_data_for_analysis(house_members_df, senate_members_df, house_votes_df, senate_votes_df):
    """
    Prepare loaded data for analysis by the CongressionalPolarizationAnalyzer
    
    Parameters:
    -----------
    house_members_df : pandas.DataFrame
        House members data
    senate_members_df : pandas.DataFrame
        Senate members data
    house_votes_df : pandas.DataFrame
        House votes data
    senate_votes_df : pandas.DataFrame
        Senate votes data
        
    Returns:
    --------
    dict
        Dictionary with prepared data for the analyzer
    """
    # Get unique members and votes
    house_members = house_members_df[['member_id', 'party', 'state', 'ideology_score']].to_dict('records')
    senate_members = senate_members_df[['member_id', 'party', 'state', 'ideology_score']].to_dict('records')
    
    house_votes = house_votes_df[['vote_id', 'member_id', 'vote', 'party']].to_dict('records')
    senate_votes = senate_votes_df[['vote_id', 'member_id', 'vote', 'party']].to_dict('records')
    
    return {
        'house_members': house_members,
        'senate_members': senate_members,
        'house_votes': house_votes,
        'senate_votes': senate_votes
    }


def integrate_with_analyzer(analyzer, data_dir):
    """
    Integrate real data with a CongressionalPolarizationAnalyzer instance
    
    Parameters:
    -----------
    analyzer : CongressionalPolarizationAnalyzer
        Analyzer instance
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    CongressionalPolarizationAnalyzer
        Updated analyzer instance with real data
    """
    # Load real data
    house_members_df, senate_members_df, house_votes_df, senate_votes_df = load_real_congressional_data(data_dir)
    
    # Prepare data for the analyzer
    prepared_data = prepare_data_for_analysis(house_members_df, senate_members_df, house_votes_df, senate_votes_df)
    
    # Update analyzer with real data
    analyzer.house_members_df = pd.DataFrame(prepared_data['house_members'])
    analyzer.senate_members_df = pd.DataFrame(prepared_data['senate_members'])
    analyzer.house_votes_df = pd.DataFrame(prepared_data['house_votes'])
    analyzer.senate_votes_df = pd.DataFrame(prepared_data['senate_votes'])
    
    print("Analyzer initialized with real congressional data.")
    
    return analyzer


# Example usage
if __name__ == "__main__":
    from congressional_polarization import CongressionalPolarizationAnalyzer
    
    # Create the analyzer
    analyzer = CongressionalPolarizationAnalyzer(congress_number=119)
    
    # Integrate real data
    data_dir = "data/congress_119"
    if os.path.exists(data_dir):
        analyzer = integrate_with_analyzer(analyzer, data_dir)
        analyzer.run_full_analysis()
    else:
        print(f"Data directory '{data_dir}' not found. Using simulated data.")
        analyzer.run_full_analysis()