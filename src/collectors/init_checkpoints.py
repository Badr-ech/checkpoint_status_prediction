"""
Initialize checkpoints database with major Palestinian checkpoints
This script populates the database with initial checkpoint data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db_context, Checkpoint, CheckpointType
from datetime import datetime


# Major checkpoints in the West Bank with real coordinates
MAJOR_CHECKPOINTS = [
    {
        "name": "Qalandiya Checkpoint",
        "name_ar": "حاجز قلنديا",
        "name_he": "מחסום קלנדיה",
        "latitude": 31.8653,
        "longitude": 35.2045,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "Between Jerusalem and Ramallah, one of the busiest checkpoints",
        "governorate": "Jerusalem/Ramallah",
        "region": "West Bank",
        "ocha_id": "CP_QALANDIYA_001"
    },
    {
        "name": "Bethlehem 300 Checkpoint",
        "name_ar": "حاجز بيت لحم 300",
        "name_he": "מחסום בית לחם 300",
        "latitude": 31.7167,
        "longitude": 35.2072,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "Main checkpoint between Bethlehem and Jerusalem",
        "governorate": "Bethlehem",
        "region": "West Bank",
        "ocha_id": "CP_BETHLEHEM_300"
    },
    {
        "name": "Huwwara Checkpoint",
        "name_ar": "حاجز حوارة",
        "name_he": "מחסום חווארה",
        "latitude": 32.1872,
        "longitude": 35.2806,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "South of Nablus, major checkpoint for northern West Bank",
        "governorate": "Nablus",
        "region": "West Bank",
        "ocha_id": "CP_HUWWARA_001"
    },
    {
        "name": "Jaba Checkpoint",
        "name_ar": "حاجز جبع",
        "name_he": "מחסום ג'בע",
        "latitude": 31.8747,
        "longitude": 35.2742,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "Northeast of Jerusalem, connects to Route 60",
        "governorate": "Jerusalem",
        "region": "West Bank",
        "ocha_id": "CP_JABA_001"
    },
    {
        "name": "Container Checkpoint",
        "name_ar": "حاجز الكونتينر",
        "name_he": "מחסום המכולה",
        "latitude": 31.7019,
        "longitude": 35.1789,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "Between Bethlehem and Hebron on Route 60",
        "governorate": "Bethlehem",
        "region": "West Bank",
        "ocha_id": "CP_CONTAINER_001"
    },
    {
        "name": "Tunnels Checkpoint",
        "name_ar": "حاجز الأنفاق",
        "name_he": "מחסום המנהרות",
        "latitude": 31.7258,
        "longitude": 35.1867,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "Main entrance to Bethlehem from the north",
        "governorate": "Bethlehem",
        "region": "West Bank",
        "ocha_id": "CP_TUNNELS_001"
    },
    {
        "name": "Za'tara Checkpoint",
        "name_ar": "حاجز زعترة",
        "name_he": "מחסום זעתרה",
        "latitude": 32.0972,
        "longitude": 35.2119,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "Between Nablus and Ramallah, major checkpoint on Route 60",
        "governorate": "Salfit",
        "region": "West Bank",
        "ocha_id": "CP_ZAATARA_001"
    },
    {
        "name": "Beit El Checkpoint",
        "name_ar": "حاجز بيت إيل",
        "name_he": "מחסום בית אל",
        "latitude": 31.9372,
        "longitude": 35.2228,
        "checkpoint_type": CheckpointType.PERMANENT,
        "location_description": "North of Ramallah, near Beit El settlement",
        "governorate": "Ramallah",
        "region": "West Bank",
        "ocha_id": "CP_BEIT_EL_001"
    }
]


def init_checkpoints():
    """Initialize database with major checkpoints"""
    with get_db_context() as db:
        # Check if checkpoints already exist
        existing_count = db.query(Checkpoint).count()
        
        if existing_count > 0:
            print(f"Database already contains {existing_count} checkpoints.")
            response = input("Do you want to add more checkpoints anyway? (y/n): ")
            if response.lower() != 'y':
                print("Initialization cancelled.")
                return
        
        print(f"\nAdding {len(MAJOR_CHECKPOINTS)} major checkpoints to the database...\n")
        
        added_count = 0
        skipped_count = 0
        
        for checkpoint_data in MAJOR_CHECKPOINTS:
            # Check if checkpoint already exists
            existing = db.query(Checkpoint).filter(
                Checkpoint.name == checkpoint_data["name"]
            ).first()
            
            if existing:
                print(f"⚠️  Skipped: {checkpoint_data['name']} (already exists)")
                skipped_count += 1
                continue
            
            # Create new checkpoint
            checkpoint = Checkpoint(**checkpoint_data)
            db.add(checkpoint)
            
            print(f"✓ Added: {checkpoint_data['name']}")
            print(f"  Location: {checkpoint_data['location_description']}")
            print(f"  Coordinates: {checkpoint_data['latitude']}, {checkpoint_data['longitude']}")
            print()
            
            added_count += 1
        
        # Commit all changes
        db.commit()
        
        print("\n" + "="*60)
        print(f"Initialization complete!")
        print(f"  Added: {added_count} checkpoints")
        print(f"  Skipped: {skipped_count} checkpoints (already existed)")
        print(f"  Total in database: {db.query(Checkpoint).count()} checkpoints")
        print("="*60)


def list_checkpoints():
    """List all checkpoints in the database"""
    with get_db_context() as db:
        checkpoints = db.query(Checkpoint).order_by(Checkpoint.name).all()
        
        if not checkpoints:
            print("No checkpoints found in database.")
            return
        
        print(f"\n{'='*80}")
        print(f"{'ID':<5} {'Name':<30} {'Location':<30} {'Type':<15}")
        print(f"{'='*80}")
        
        for cp in checkpoints:
            print(f"{cp.id:<5} {cp.name:<30} {cp.governorate or 'N/A':<30} {cp.checkpoint_type.value:<15}")
        
        print(f"{'='*80}")
        print(f"Total: {len(checkpoints)} checkpoints\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize checkpoint database")
    parser.add_argument("--list", action="store_true", help="List existing checkpoints")
    parser.add_argument("--init", action="store_true", help="Initialize checkpoints")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.init:
        init_checkpoints()
    else:
        # Default: initialize
        print("Checkpoint Database Initialization")
        print("=" * 60)
        init_checkpoints()
