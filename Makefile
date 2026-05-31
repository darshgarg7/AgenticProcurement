.PHONY: test unittest lint compile quality

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

unittest:
	python3 -m unittest discover -s tests

lint:
	python3 -m ruff check .

compile:
	python3 -m compileall -q config core decision environment evaluation experiments models web_demo tests

quality: lint test compile
