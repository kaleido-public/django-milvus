default:
	make '*'

report:
	coverage report --skip-empty --skip-covered


bail:
	coverage run --branch --source=../../ ../../manage.py test --no-input -v2 -p "*.py" --failfast
	coverage xml

parallel:
	coverage run --branch --source=../../ ../../manage.py test --no-input -v2 -p "*.py" --parallel
	coverage xml

target %: argv = $(subst ., ,$@)
target %: module = $(word 1, $(argv))
target %: method = $(word 2, $(argv))
%:
	coverage run --source=../../ ../../manage.py test --no-input -v2 -p "$(module).py" -k "$(method)"
	coverage xml
