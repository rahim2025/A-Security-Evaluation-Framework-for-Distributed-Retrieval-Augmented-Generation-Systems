#!/bin/bash

echo "=========================================="
echo "Testing Knowledge Base Extraction Attack"
echo "=========================================="

# Run with test configuration - all settings are in the YAML
python3 simulator.py --config config/test_extraction.yaml

echo ""
echo "=========================================="
echo "Test Complete - Check logs/version_X/"
echo "=========================================="