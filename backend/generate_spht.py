#!/usr/bin/env python3
"""
Generate SPHT (Star Pattern Hash Table) for star identification
"""

from helper_functions import get_star_catalog, build_spht_offline, save_spht_to_json
import random

def generate_spht():
    print("Loading star catalog...")
    bsc = get_star_catalog()
    
    # Use a subset for faster processing (you can increase this for better accuracy)
    print("Creating subset of star catalog...")
    random.seed(42)
    subset_bsc = random.sample(bsc, 200)  # Use 200 stars for good coverage
    
    print(f"Building SPHT from {len(subset_bsc)} stars...")
    al_parameter = 1
    spht = build_spht_offline(subset_bsc, al_parameter)
    
    print("Saving SPHT to JSON...")
    save_spht_to_json(spht, "spht.json")
    
    print(f"SPHT generated successfully with {len(spht)} entries!")

if __name__ == "__main__":
    generate_spht() 