#!/bin/bash


# Run with a custom sample size:
#    ./testing.sh 200

SAMPLE_SIZE=${1:-50}

echo "RAG Benchmark"
echo "Sample Size: $SAMPLE_SIZE"

# Run the benchmark python module
python -m src.evaluation.benchmark $SAMPLE_SIZE


