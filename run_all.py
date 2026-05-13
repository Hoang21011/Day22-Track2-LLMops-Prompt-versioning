import argparse
import subprocess
import sys
import os

def run_step(step_number):
    scripts = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }
    
    script = scripts.get(step_number)
    if not script:
        print(f"❌ Unknown step: {step_number}")
        return

    print(f"\n🚀 Running Step {step_number}: {script}...")
    
    # Use tee to log output if it's step 2 or 4 (as requested in README)
    log_files = {
        2: "evidence/02_ab_routing_log.txt",
        4: "evidence/04_pii_demo_log.txt" # This is just one of them
    }
    
    try:
        if step_number in log_files:
            log_path = log_files[step_number]
            # Use shell to pipe to tee
            process = subprocess.Popen(
                f"{sys.executable} {script} | tee {log_path}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in process.stdout:
                print(line, end='')
            process.wait()
        else:
            subprocess.run([sys.executable, script], check=True)
            
        print(f"✅ Step {step_number} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Step {step_number} failed with error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab Steps")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a specific step")
    args = parser.parse_args()

    if args.step:
        run_step(args.step)
    else:
        # Run all steps
        for i in range(1, 5):
            run_step(i)
            print("-" * 40)

if __name__ == "__main__":
    main()
