SBATCH HEADERS HERE

# inscructions for running on pace_ice, we'll worry about this later

# This is the first argument passed to the script, which is the number of iterations
# Usage: sbatch pace_ice_island_controller.sbatch <number_of_iterations>
# Example: sbatch pace_ice_island_controller.sbatch 10
# This is a workaround for pace_ice time limits
COUNT=$1

echo "$COUNT iterations remaining"

WHATEVER YOU WANT TO RUN (ENSURE IT'LL FINISH RUNNING BEFORE PACE ICE LIMIT)

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    echo "Launching next iteration with count: $NEXT_COUNT"
    uv run sbatch pace_ice_island_controller.sbatch "$NEXT_COUNT"
fi