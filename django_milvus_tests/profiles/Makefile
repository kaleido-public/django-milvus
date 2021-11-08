target %: argv = $(subst ., ,$@)
target %: module = $(word 1, $(argv))
target %: method = $(word 2, $(argv))
%:
	rm -f cprofile
	../../manage.py test --no-input -v2 -p "$(module).py" -k "$(method)"
	snakeviz cprofile
