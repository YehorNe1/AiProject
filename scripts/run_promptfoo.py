"""
Script to run all Promptfoo experiments defined in promptfoo.yml.
"""

import subprocess
import sys

def main():
    # Execute Promptfoo using 'eval' command and our config file
    result = subprocess.run(
        ["promptfoo", "eval", "--config", "promptfoo.yml"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("❌ Promptfoo tests failed", file=sys.stderr)
        sys.exit(result.returncode)
    print("✅ All Promptfoo tests passed")

if __name__ == "__main__":
    main()
