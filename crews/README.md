# Flows

## Setup

Navigate to the flows directory:

```bash
cd flows
```

Create a .env file and add your desired model api keys

### Option 1: Manual Setup

1. Create a virtual environment:

    ```bash
    python3 -m venv .venv
    ```

2. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install .
    ```

### Option 2: Using CrewAI CLI (Recommended)

If you have CrewAI CLI installed, you can simply run:

```bash
crewai install
```

## Running the Application

To run the travel flow, you can use either of these commands:

```bash
python -m src.testing_crews.main
```

Or alternatively:

```bash
python src/testing_crews/main.py
```
