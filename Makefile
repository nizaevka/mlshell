init:
	pip install pipenv
	pipenv install pytest  #--dev

test:
	pipenv run pytest tests