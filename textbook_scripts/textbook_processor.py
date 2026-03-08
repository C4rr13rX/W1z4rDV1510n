import subprocess
import sys
import os
import argparse

def run_textbook_processing(test_mode=True):
    """Run textbook download and processing with safety controls"""
    
    if test_mode:
        print("[TEST MODE] Textbook processor initialized")
        print("[TEST MODE] Would check Node.js availability")
        print("[TEST MODE] Would verify textbook scripts")
        print("[TEST MODE] Would process textbooks with limits")
        return True
    
    try:
        # Check Node.js availability
        result = subprocess.run(["node", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("Error: Node.js not available")
            return False
            
        print(f"Node.js version: {result.stdout.strip()}")
        
        # Run textbook processing script
        script_path = os.path.join("textbook_scripts", "download_and_process.mjs")
        if os.path.exists(script_path):
            result = subprocess.run(["node", script_path], 
                                  capture_output=True, text=True, timeout=300)
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            return result.returncode == 0
        else:
            print(f"Error: Script not found: {script_path}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Error: Processing timeout")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Textbook processor")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--full", action="store_true", help="Run full processing")
    
    args = parser.parse_args()
    
    if args.test:
        success = run_textbook_processing(test_mode=True)
    elif args.full:
        success = run_textbook_processing(test_mode=False)
    else:
        print("Usage: python textbook_processor.py --test | --full")
        success = False
    
    sys.exit(0 if success else 1)
