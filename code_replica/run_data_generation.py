"""
run_data_generation.py

Script to execute the data generation and testing process.
"""

import subprocess
import sys
import os

def run_script(script_name: str, description: str):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"Script {script_name} not found.")
        return False

def main():
    """Main execution function."""
    print("CoMET Synthetic Data Generation and Testing Pipeline")
    
    # Step 1: Generate synthetic data
    success = run_script("generate_synthetic_data.py", "Generating Synthetic Medical Data")
    if not success:
        print("❌ Data generation failed. Stopping.")
        return
    
    # Step 2: Test compatibility (only if test script exists)
    if os.path.exists("test_data_compatibility.py"):
        success = run_script("test_data_compatibility.py", "Testing Data Compatibility")
        if not success:
            print("⚠️  Compatibility tests failed, but data was generated successfully.")
    else:
        print("⚠️  test_data_compatibility.py not found. Skipping compatibility tests.")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED")
    print(f"{'='*60}")
    print("✅ Synthetic medical data has been generated successfully!")
    print("📁 Check the 'data/' directory for the generated files.")
    print("🔬 Data is ready for CoMET model training and evaluation.")

if __name__ == "__main__":
    main()