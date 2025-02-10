# venv, install requirements, dev-reequirements

.PHONY: venv install install-dev pre-commit clean

.venv:
	python3 -m venv .venv

venv: .venv

install: .venv
	. .venv/bin/activate; pip install -r requirements.txt

install-dev: .venv
	. .venv/bin/activate; pip install -r requirements-dev.txt

pre-commit: install-dev
	. .venv/bin/activate; pre-commit install

clean:
	rm -rf .venv
	find . -name "*.pyc" -exec rm -f {} \;
	find . -name "__pycache__" -exec rm -rf {} \;
