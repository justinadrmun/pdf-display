# Justfile for pdf-display project

# Usage: just <recipe>

# Install all dependencies using uv and pyproject.toml

# Create a uv virtual environment and install dependencies
init:
    uv venv
    uv pip install -r pyproject.toml

# Activate the uv virtual environment (for manual use)
activate:
    source .venv/bin/activate

# Run the benchmark app

# Run the benchmark app using the venv
benchmark:
    .venv/bin/streamlit run benchmark_app.py
