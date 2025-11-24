#!/usr/bin/env python3
"""
Challenge runner script for reframe versions.
Usage: 
    python run_test.py --reframe_scripts_folder_name 1700000000 --challenges_folder_name 1700000000
    python run_test.py --reframe_scripts_folder_name 1700000000 --challenges_folder_name all
"""
import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

def find_reframe_script(folder_name):
    """Find reframe script by folder name."""
    reframe_dir = Path("reframe_scripts")
    if not reframe_dir.exists():
        return None
    
    script_dir = reframe_dir / folder_name
    if not script_dir.exists() or not script_dir.is_dir():
        return None
    
    # Find any Python script file in the folder (name doesn't matter)
    python_files = [f for f in script_dir.iterdir() if f.is_file() and f.suffix == ".py"]
    
    if not python_files:
        return None
    
    # Use the first Python file found (or prefer reframe.py if it exists)
    script_file = None
    for file in python_files:
        if file.name == "reframe.py":
            script_file = file
            break
    
    if not script_file:
        script_file = python_files[0]  # Use first .py file found
    
    return {
        'folder_name': folder_name,
        'path': script_file,
        'name': script_file.stem,
        'dir': script_dir
    }

def find_challenges():
    """Find all challenge directories."""
    challenges_dir = Path("challenges")
    if not challenges_dir.exists():
        return []
    
    challenges = []
    for challenge_dir in sorted(challenges_dir.iterdir()):
        if challenge_dir.is_dir():
            # Find the video file and requirements file
            video_file = None
            requirements_file = None
            
            for file in challenge_dir.iterdir():
                if file.is_file():
                    if file.suffix == ".mp4":
                        video_file = file
                    elif file.name in ["requirements.txt", "requirements.md"]:
                        requirements_file = file
            
            if video_file:
                challenges.append({
                    'folder_name': challenge_dir.name,
                    'path': challenge_dir,
                    'video': video_file,
                    'requirements': requirements_file
                })
    
    return sorted(challenges, key=lambda x: x['folder_name'])

def find_challenge(folder_name):
    """Find a specific challenge by folder name."""
    if folder_name.lower() == "all":
        return find_challenges()
    
    challenges_dir = Path("challenges")
    if not challenges_dir.exists():
        return []
    
    challenge_dir = challenges_dir / folder_name
    if not challenge_dir.exists() or not challenge_dir.is_dir():
        return []
    
    # Find the video file and requirements file
    video_file = None
    requirements_file = None
    
    for file in challenge_dir.iterdir():
        if file.is_file():
            if file.suffix == ".mp4":
                video_file = file
            elif file.name in ["requirements.txt", "requirements.md"]:
                requirements_file = file
    
    if video_file:
        return [{
            'folder_name': folder_name,
            'path': challenge_dir,
            'video': video_file,
            'requirements': requirements_file
        }]
    
    return []

def run_challenge(script_info, challenge_info, preview=False):
    """Run a single challenge."""
    script_path = script_info['path']
    script_folder_name = script_info['folder_name']
    challenge_dir = challenge_info['path']
    video_file = challenge_info['video']
    challenge_folder_name = challenge_info['folder_name']
    requirements_file = challenge_info.get('requirements')
    
    if not video_file.exists():
        print(f"⚠ Warning: Video file not found: {video_file}, skipping...")
        return False

    # Create output directory structure: test_results/reframe_script_{folder_name}/challenge_{folder_name}/
    output_dir = Path(f"test_results/reframe_script_{script_folder_name}/challenge_{challenge_folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file path: use the video file name
    video_name = video_file.stem  # filename without extension
    output_file = output_dir / f"{video_name}.mp4"

    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"Reframe Script Folder: {script_folder_name}")
    print(f"Challenge Folder: {challenge_folder_name}")
    print(f"Input: {video_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    # Build command
    cmd = [sys.executable, str(script_path), str(video_file), "--output", str(output_file), "--cleanup"]
    if preview:
        cmd.append("--preview")

    # Run the script
    try:
        result = subprocess.run(cmd, check=True, cwd=Path.cwd())
        
        # Verify output file was created
        if output_file.exists():
            print(f"✓ Success! Output saved to: {output_file}")
            
            # Copy requirements file if it exists
            if requirements_file and requirements_file.exists():
                # Determine the extension (.txt or .md)
                req_ext = requirements_file.suffix
                req_output = output_dir / f"requirements{req_ext}"
                shutil.copy2(requirements_file, req_output)
                print(f"✓ Requirements file copied to: {req_output}")
            
            return True
        else:
            print(f"⚠ Warning: Output file not found at expected location: {output_file}")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running script: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run reframe script on challenges")
    parser.add_argument("--reframe_scripts_folder_name", type=str, required=True,
                       help="Folder name in reframe_scripts/ (e.g., 1700000000)")
    parser.add_argument("--challenges_folder_name", type=str, required=True,
                       help="Folder name in challenges/ or 'all' to run all challenges")
    parser.add_argument("--preview", action="store_true", help="Show preview window")
    args = parser.parse_args()

    # Find reframe script
    script_info = find_reframe_script(args.reframe_scripts_folder_name)
    if not script_info:
        print(f"Error: Reframe script folder not found: {args.reframe_scripts_folder_name}")
        print(f"Expected path: reframe_scripts/{args.reframe_scripts_folder_name}/")
        sys.exit(1)

    # Find challenges
    challenge_list = find_challenge(args.challenges_folder_name)
    if not challenge_list:
        if args.challenges_folder_name.lower() == "all":
            print("Error: No challenges found in challenges/ directory")
        else:
            print(f"Error: Challenge folder not found: {args.challenges_folder_name}")
            print(f"Expected path: challenges/{args.challenges_folder_name}/")
        sys.exit(1)

    if args.challenges_folder_name.lower() == "all":
        print(f"Found {len(challenge_list)} challenge(s)")

    # Run challenges
    results = []
    for challenge_info in challenge_list:
        try:
            success = run_challenge(script_info, challenge_info, args.preview)
            results.append((challenge_info['folder_name'], success))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    for folder_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} Challenge {folder_name}")
    print(f"\nCompleted: {successful}/{total} successful")
    print(f"{'='*60}")
    
    if successful < total:
        sys.exit(1)

if __name__ == "__main__":
    main()
