# Create virtual environment
python -m venv venv

# Windows git bash
source venv/Scripts/activate

# macOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Save dependencies
pip freeze > requirements.txt

# Deactivate
deactivate

# Running tests
source venv/Scripts/activate
pip install -r requirements.txt
bash run_tests.sh           
