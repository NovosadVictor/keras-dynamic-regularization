make_env:
	virtualenv venv

install:
	pip install --upgrade pip
	pip install -r requirements.txt
