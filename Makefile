.PHONY: venv install install-dev pre-commit clean

uv:
	curl --proto '=https' --tlsv1.2 -LsSf https://github.com/astral-sh/uv/releases/download/0.5.31/uv-installer.sh | sh

.venv:
	uv venv -p 3.11 .venv --seed
	. .venv/bin/activate; pip install -r requirements-setup.txt

venv: .venv

install: .venv
	. .venv/bin/activate; pip install -r requirements.txt

install-dev: .venv
	. .venv/bin/activate; pip install -r requirements-dev.txt

install-pre-commit: install-dev
	. .venv/bin/activate; pre-commit install

clean:
	rm -rf .venv
	find . -name "*.pyc" -exec rm -f {} \;
	find . -name "__pycache__" -exec rm -rf {} \;
