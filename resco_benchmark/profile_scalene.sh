set -o allexport
source .env
set +o allexport
python -m scalene --html --outfile scalene.html main.py --agent IDQN --map BB5B --eps 3