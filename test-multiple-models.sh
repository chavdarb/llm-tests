#!/bin/bash
set -e

# Default values
OUTPUT_FILE="benchmark-results.txt"
RUNS=5
WARMUP=2
NUM_PREDICT=8192
NUM_CTX=4096
PROMPT="Explain TCP slow start in 150 tokens."
PROMPT_FILE=""

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $0 [OPTIONS] model1 model2 model3 ...

Test multiple Ollama models and output results to a file.

OPTIONS:
    -o, --output FILE       Output file name (default: benchmark-results.txt)
    -r, --runs N           Number of test runs per model (default: 5)
    -w, --warmup N         Number of warmup runs (default: 2)
    --prompt TEXT          Prompt text to use
    --prompt-file FILE     Read prompt from file
    --num-predict N        Max tokens to generate (default: 8192)
    --num-ctx N            Context window size (default: 4096)
    -h, --help             Show this help message

EXAMPLES:
    $0 -o results.txt llama3.2:3b qwen3:8b mistral:7b
    $0 --runs 10 --prompt "Write a poem" llama3.2:3b qwen3:8b
    $0 --prompt-file prompts.txt -o my-results.txt llama3.2:3b

EOF
    exit 1
}

# Parse arguments
MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -r|--runs)
            RUNS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --num-predict)
            NUM_PREDICT="$2"
            shift 2
            ;;
        --num-ctx)
            NUM_CTX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done

# Check if models are specified
if [ ${#MODELS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No models specified${NC}"
    usage
fi

# Check if ollama-bench.sh exists
if [ ! -f "ollama-bench.sh" ]; then
    echo -e "${RED}Error: ollama-bench.sh not found in current directory${NC}"
    exit 1
fi

# Make ollama-bench.sh executable if it isn't
chmod +x ollama-bench.sh

# Prepare output file
{
    echo "=============================================="
    echo "Ollama Model Benchmark Results"
    echo "=============================================="
    echo "Date: $(date)"
    echo "Models: ${MODELS[*]}"
    echo "Runs per model: $RUNS"
    echo "Warmup runs: $WARMUP"
    echo "Max tokens: $NUM_PREDICT"
    echo "Context window: $NUM_CTX"
    if [ -n "$PROMPT_FILE" ]; then
        echo "Prompt file: $PROMPT_FILE"
    else
        echo "Prompt: $PROMPT"
    fi
    echo "=============================================="
    echo ""
} > "$OUTPUT_FILE"

# Build base command arguments
BENCH_ARGS="--runs $RUNS --warmup $WARMUP --num-predict $NUM_PREDICT --num-ctx $NUM_CTX"
if [ -n "$PROMPT_FILE" ]; then
    BENCH_ARGS="$BENCH_ARGS --prompt-file $PROMPT_FILE"
else
    BENCH_ARGS="$BENCH_ARGS --prompt \"$PROMPT\""
fi

# Test each model
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0

for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo -e "${YELLOW}[${CURRENT}/${TOTAL_MODELS}] Testing model: ${model}${NC}"
    
    {
        echo "=============================================="
        echo "Model: $model"
        echo "=============================================="
    } >> "$OUTPUT_FILE"
    
    # Run benchmark and append to file
    if eval "./ollama-bench.sh --model \"$model\" $BENCH_ARGS" >> "$OUTPUT_FILE" 2>&1; then
        echo -e "${GREEN}✓ Model ${model} completed successfully${NC}"
    else
        echo -e "${RED}✗ Model ${model} failed${NC}"
        echo "ERROR: Benchmark failed for model $model" >> "$OUTPUT_FILE"
    fi
    
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

# Final summary
{
    echo "=============================================="
    echo "Benchmark completed at $(date)"
    echo "=============================================="
} >> "$OUTPUT_FILE"

echo -e "${GREEN}All benchmarks completed!${NC}"
echo -e "Results saved to: ${YELLOW}${OUTPUT_FILE}${NC}"
