default:
	$(MAKE) '*'

report:
	coverage report --skip-empty --skip-covered


bail:
	$(MAKE) reset-postgres
	coverage run --branch --source=../../ ../../manage.py test --no-input -v2 -p "*.py" --failfast
	coverage xml

parallel:
	$(MAKE) reset-postgres
	coverage run --branch --source=../../ ../../manage.py test --no-input -v2 -p "*.py" --parallel
	coverage xml

target %: argv = $(subst ., ,$@)
target %: module = $(word 1, $(argv))
target %: method = $(word 2, $(argv))
%:
	$(MAKE) reset-postgres
	coverage run --source=../../ ../../manage.py test --no-input -v2 -p "$(module).py" -k "$(method)"
	coverage xml


reset-postgres:
	docker-compose up --force-recreate -V -d
	sleep 1
	wait-for-it localhost:5333
