#!/bin/bash
set -e

# Define all movies and their stimulus subtypes
declare -A STIMS=(
    ["friends"]="s1 s2 s3 s4 s5 s6 s7"
    ["movie10"]="bourne figures life wolf"
    ["ood"]="chaplin mononoke passepartout planetearth pulpfiction wot"
)

# Loop through all combinations and run the extraction
for movie in "${!STIMS[@]}"; do
    for stim in ${STIMS[$movie]}; do
        echo "========================================================="
        echo "Extracting VLM2Vec features for: $movie / $stim"
        echo "========================================================="
        python src/data/features/extract_vlm2vec.py \
            --movie_type "$movie" \
            --stimulus_type "$stim"
    done
done

echo "🎉 All features extracted!"
