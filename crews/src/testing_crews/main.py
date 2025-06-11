#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime

from testing_crews.crew import TestingCrews


def run():
    """
    Run the crew.
    """
    query = input("Enter your trip query: ")
    inp = {
        "query": query
    }

    print(inp)
    
    try:
        TestingCrews().crew().kickoff(inputs=inp)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()   