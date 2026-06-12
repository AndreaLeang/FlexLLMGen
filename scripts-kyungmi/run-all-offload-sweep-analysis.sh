#!/bin/bash

# Usage: ./analyze_traces.sh <output_folder>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_folder>"
    exit 1
fi

OUTPUT_DIR="$1"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Directory not found: ${OUTPUT_DIR}"
    exit 1
fi

for SUBDIR in "${OUTPUT_DIR}"/prompt_*_bs*/; do
    if [ ! -d "$SUBDIR" ]; then
        continue
    fi

    echo "=== Processing: ${SUBDIR} ==="

    for TRACE in "${SUBDIR}"*.json; do
        if [ ! -f "$TRACE" ]; then
            echo "  No JSON files found in ${SUBDIR}, skipping."
            continue
        fi

        STEM=$(python3 -c "from pathlib import Path; print(Path('${TRACE}').stem)")
        CSV_FILE="${SUBDIR}${STEM}_nosep_analysis.csv"

        echo "  Analyzing trace: ${TRACE}"
        python3 trace_analyzer.py "${TRACE}" --nosep

        if [ $? -ne 0 ]; then
            echo "  ERROR: trace_analyzer.py failed for ${TRACE}. Skipping result analysis."
            continue
        fi

        echo "  Analyzing results: ${CSV_FILE}"
        python3 trace_result_analyzer.py "${CSV_FILE}" --nosep

        if [ $? -ne 0 ]; then
            echo "  ERROR: trace_result_analyzer.py failed for ${CSV_FILE}."
        else
            echo "  Done: ${TRACE}"
        fi

    done

done

echo "=== All analysis complete ==="