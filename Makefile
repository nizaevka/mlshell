test:
	pip install --user pytest
	pytest tests
deploy:
	pip install --user --upgrade setuptools wheel
	python setup.py sdist bdist_wheel
	pip install --user --upgrade twine
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* -u $(TEST_PYPI_USERNAME) -p $(TEST_PYPI_PASSWORD)