echo "Analyzing all experiment results..."

for size in small medium large xlarge; do
    if [ -f "data/profiling_${size}.csv" ]; then
        echo "Analyzing ${size}..."
        python3 python/analyze_results.py \
            -i "data/profiling_${size}.csv" \
            -o "data/analysis_${size}"
    fi
done

echo ""
echo "Comparing all experiments..."
python3 python/compare_experiments.py

echo ""
echo "✓ All analysis completed!"