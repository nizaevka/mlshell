test:
	pip install pytest
	pytest tests
deploy:
	pip install --upgrade setuptools wheel
	python setup.py sdist bdist_wheel
	pip install --upgrade twine
	python -m twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/* -u $(TEST_PYPI_USERNAME) -p $(TEST_PYPI_PASSWORD)