set -o allexport
source .env
set +o allexport
python -m cProfile -s cumulative -o cprofile.stats main.py --agent IDQN --map BB5B --eps 3