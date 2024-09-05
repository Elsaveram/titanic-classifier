#!/bin/sh

# This script runs the Titanic pipeline and associated tests.

# Set PYTHONPATH to point to the src folder so that modules can be imported correctly
export PYTHONPATH=src

# Directory where the configuration and test files are located
CONFIG_FILE=config/titanic_config.json
TEST_DIR=tests

# Step 1: Run the Titanic Pipeline
echo "Step 1: Running Titanic Pipeline..."
poetry run python3 src/titanic_pipeline.py --config $CONFIG_FILE

# Check if the pipeline ran successfully
if [ $? -ne 0 ]; then
  echo "Pipeline execution failed. Exiting..."
  exit 1
fi
echo "Pipeline executed successfully."

# Step 2: Run Unit Tests
echo "Step 2: Running Unit Tests..."
poetry run pytest --maxfail=5 --disable-warnings $TEST_DIR/test_titanic_preprocessing.py

# Check if tests ran successfully
if [ $? -ne 0 ]; then
  echo "Some tests failed. Please check the output above."
  exit 1
fi

echo "All tests passed successfully!"

# Exit with success status
exit 0
