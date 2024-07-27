# main.py for Paper1_Active_Inference_Hebbian

import subprocess

def run_all_figures():
    # Call the script for Figures 10, 14, and 15
    subprocess.run(["python", "Figure_10_14_15.py"])
    
    # Call the script for Figure 11
    subprocess.run(["python", "Figure_11.py"])
    
    # Call the script for Figure 12
    subprocess.run(["python", "Figure_12.py"])
    
    # Call the script for Figure 13
    subprocess.run(["python", "Figure_13.py"])

if __name__ == "__main__":
    run_all_figures()

