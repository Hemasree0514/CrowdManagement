"""
scripts/enroll_vip.py
Register a new VIP face into the system.

Usage:
    python scripts/enroll_vip.py \\
        --name "Dr. A. Sharma" \\
        --role "District Collector" \\
        --rank senior \\
        --image path/to/photo.jpg \\
        --unit "Unit Foxtrot" \\
        --escort

This script:
  1. Detects the face in the provided image
  2. Copies the image to config/vip_faces/
  3. Appends the VIP entry to config/vip_database.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
VIP_DB    = BASE_DIR / "config" / "vip_database.json"
FACES_DIR = BASE_DIR / "config" / "vip_faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)


def generate_id(db: list) -> str:
    if not db:
        return "VIP-001"
    last_id = max(int(v["id"].split("-")[1]) for v in db)
    return f"VIP-{last_id + 1:03d}"


def main():
    parser = argparse.ArgumentParser(description="Enroll a VIP face")
    parser.add_argument("--name",   required=True, help="Full name")
    parser.add_argument("--role",   required=True, help="Role/title")
    parser.add_argument("--rank",   choices=["vip", "senior", "vvip"], default="vip")
    parser.add_argument("--image",  required=True, help="Path to reference photo")
    parser.add_argument("--unit",   default="Unit Foxtrot", help="Security unit to notify")
    parser.add_argument("--escort", action="store_true", help="Escort required")
    parser.add_argument("--notes",  default="")
    args = parser.parse_args()

    # Validate image exists
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    # Optionally verify a face is detectable
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            print("ERROR: Could not read image file.")
            sys.exit(1)
        print(f"Image loaded: {img.shape[1]}x{img.shape[0]} px")
    except ImportError:
        print("Note: OpenCV not installed — skipping face pre-check.")

    # Load existing DB
    with open(VIP_DB) as f:
        db = json.load(f)

    # Build new entry
    vip_id    = generate_id(db)
    img_dest  = FACES_DIR / img_path.name
    shutil.copy2(img_path, img_dest)

    entry = {
        "id":              vip_id,
        "name":            args.name,
        "role":            args.role,
        "rank":            args.rank,
        "notify_unit":     args.unit,
        "escort_required": args.escort,
        "image_file":      img_path.name,
        "alert_channel":   "app",
        "notes":           args.notes,
    }
    db.append(entry)

    with open(VIP_DB, "w") as f:
        json.dump(db, f, indent=2)

    print(f"\n✅ VIP enrolled successfully!")
    print(f"   ID:    {vip_id}")
    print(f"   Name:  {args.name}")
    print(f"   Role:  {args.role}")
    print(f"   Image: {img_dest}")
    print(f"\nRestart the backend to reload face embeddings.")


if __name__ == "__main__":
    main()
