#!/usr/bin/env python3
"""
Congressional Data Fetcher

This script fetches real congressional voting data using the ProPublica Congress API
and prepares it for analysis. It's an alternative to the simulated data in the main
analysis script.

Usage:
    python data_fetcher.py --api-key YOUR_API_KEY --congress 119 --output data/congress_119

Note:
    You need a ProPublica Congress API key, which you can get at:
    https://www.propublica.org/datastore/api/propublica-congress-api
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm


class CongressDataFetcher:
    def __init__(self, api_key, congress_number=119):
        """
        Initialize the data fetcher
        
        Parameters:
        -----------
        api_key : str
            ProPublica API key
        congress_number : int
            Congress number to fetch (default: 119)
        """
        self.api_key = api_key
        self.congress_number = congress_number
        self.base_url = "https://api.propublica.org/congress/v1"
        self.headers = {"X-API-Key": api_key}
        
        # Data storage
        self.house_members = []
        self.senate_members = []
        self.house_votes = []
        self.senate_votes = []
    
    def fetch_members(self):
        """Fetch members of the House and Senate"""
        print(f"Fetching members of the {self.congress_number}th Congress...")
        
        # Fetch House members
        house_url = f"{self.base_url}/{self.congress_number}/house/members.json"
        response = requests.get(house_url, headers=self.headers)
        
        if response.status_code == 200:
            house_data = response.json()
            self.house_members = house_data['results'][0]['members']
            print(f"Retrieved {len(self.house_members)} House members.")
        else:
            print(f"Error fetching House members: {response.status_code}")
            print(response.text)
        
        # Fetch Senate members
        senate_url = f"{self.base_url}/{self.congress_number}/senate/members.json"
        response = requests.get(senate_url, headers=self.headers)
        
        if response.status_code == 200:
            senate_data = response.json()
            self.senate_members = senate_data['results'][0]['members']
            print(f"Retrieved {len(self.senate_members)} Senate members.")
        else:
            print(f"Error fetching Senate members: {response.status_code}")
            print(response.text)
    
    def fetch_vote_data(self, chamber='house', num_votes=100):
        """
        Fetch voting data for a chamber
        
        Parameters:
        -----------
        chamber : str
            'house' or 'senate' (default: 'house')
        num_votes : int
            Number of most recent votes to fetch (default: 100)
        """
        print(f"Fetching {num_votes} recent votes for the {chamber.capitalize()}...")
        votes = []
        
        # Get recent votes
        recent_votes_url = f"{self.base_url}/{self.congress_number}/{chamber}/votes/recent.json"
        response = requests.get(recent_votes_url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Error fetching recent votes: {response.status_code}")
            print(response.text)
            return
        
        recent_votes = response.json()['results']['votes']
        vote_ids = [vote['roll_call'] for vote in recent_votes[:num_votes]]
        
        # Fetch each vote's details
        for vote_id in tqdm(vote_ids, desc=f"Fetching {chamber.capitalize()} votes"):
            vote_url = f"{self.base_url}/{self.congress_number}/{chamber}/sessions/1/votes/{vote_id}.json"
            response = requests.get(vote_url, headers=self.headers)
            
            if response.status_code == 200:
                vote_data = response.json()['results']['votes']['vote']
                votes.append(vote_data)
                
                # Respect API rate limits
                time.sleep(0.2)
            else:
                print(f"Error fetching vote {vote_id}: {response.status_code}")
        
        if chamber == 'house':
            self.house_votes = votes
        else:
            self.senate_votes = votes
        
        print(f"Retrieved {len(votes)} {chamber.capitalize()} votes.")
    
    def process_member_data(self):
        """Process and clean member data into DataFrames"""
        print("Processing member data...")
        
        # Process House members
        house_df = pd.DataFrame(self.house_members)
        house_df = house_df[[
            'id', 'first_name', 'last_name', 'party', 'state', 
            'district', 'next_election', 'total_votes', 'missed_votes', 
            'total_present', 'votes_with_party_pct'
        ]]
        
        # Process Senate members
        senate_df = pd.DataFrame(self.senate_members)
        senate_df = senate_df[[
            'id', 'first_name', 'last_name', 'party', 'state', 
            'next_election', 'total_votes', 'missed_votes', 
            'total_present', 'votes_with_party_pct'
        ]]
        
        return house_df, senate_df
    
    def process_vote_data(self):
        """Process voting data into a format suitable for analysis"""
        print("Processing voting data...")
        
        # Process House votes
        house_votes_processed = []
        for vote in self.house_votes:
            vote_id = vote['roll_call']
            
            # Skip votes with no position data
            if 'positions' not in vote:
                continue
                
            for position in vote['positions']:
                house_votes_processed.append({
                    'vote_id': vote_id,
                    'member_id': position['member_id'],
                    'vote_position': position['vote_position'],
                    'party': position['party']
                })
        
        # Process Senate votes
        senate_votes_processed = []
        for vote in self.senate_votes:
            vote_id = vote['roll_call']
            
            # Skip votes with no position data
            if 'positions' not in vote:
                continue
                
            for position in vote['positions']:
                senate_votes_processed.append({
                    'vote_id': vote_id,
                    'member_id': position['member_id'],
                    'vote_position': position['vote_position'],
                    'party': position['party']
                })
        
        house_votes_df = pd.DataFrame(house_votes_processed)
        senate_votes_df = pd.DataFrame(senate_votes_processed)
        
        # Convert vote positions to binary (Yes/No)
        for df in [house_votes_df, senate_votes_df]:
            df['vote'] = df['vote_position'].map({
                'Yes': 'Yes',
                'No': 'No',
                'Present': 'No',
                'Not Voting': None
            })
            
        return house_votes_df, senate_votes_df
    
    def save_data(self, output_dir):
        """
        Save fetched data to CSV files
        
        Parameters:
        -----------
        output_dir : str
            Directory to save data files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process and save member data
        house_members_df, senate_members_df = self.process_member_data()
        house_members_df.to_csv(os.path.join(output_dir, 'house_members.csv'), index=False)
        senate_members_df.to_csv(os.path.join(output_dir, 'senate_members.csv'), index=False)
        
        # Process and save vote data
        house_votes_df, senate_votes_df = self.process_vote_data()
        house_votes_df.to_csv(os.path.join(output_dir, 'house_votes.csv'), index=False)
        senate_votes_df.to_csv(os.path.join(output_dir, 'senate_votes.csv'), index=False)
        
        # Save raw data as JSON
        with open(os.path.join(output_dir, 'raw_house_members.json'), 'w') as f:
            json.dump(self.house_members, f)
        
        with open(os.path.join(output_dir, 'raw_senate_members.json'), 'w') as f:
            json.dump(self.senate_members, f)
        
        with open(os.path.join(output_dir, 'raw_house_votes.json'), 'w') as f:
            json.dump(self.house_votes, f)
        
        with open(os.path.join(output_dir, 'raw_senate_votes.json'), 'w') as f:
            json.dump(self.senate_votes, f)
        
        print(f"Data saved to {output_dir}/")
    
    def fetch_all_data(self, num_votes=100):
        """
        Fetch all data (members and votes)
        
        Parameters:
        -----------
        num_votes : int
            Number of votes to fetch for each chamber (default: 100)
        """
        self.fetch_members()
        self.fetch_vote_data(chamber='house', num_votes=num_votes)
        self.fetch_vote_data(chamber='senate', num_votes=num_votes)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fetch Congressional Data')
    parser.add_argument('--api-key', type=str, required=True, help='ProPublica API Key')
    parser.add_argument('--congress', type=int, default=119, help='Congress number')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--votes', type=int, default=100, help='Number of votes to fetch per chamber')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output, f'congress_{args.congress}')
    
    # Fetch data
    fetcher = CongressDataFetcher(args.api_key, args.congress)
    fetcher.fetch_all_data(num_votes=args.votes)
    fetcher.save_data(output_dir)


if __name__ == "__main__":
    main()