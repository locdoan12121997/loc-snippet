alias sv='source .venv/bin/activate'
alias ga='git add .'
alias gac='git add . && git commit -m'
alias gp='git push origin master'
alias pir='pip install -r requirements.txt'
alias vir='virtualenv -p python3 .venv'
alias mnt="mount | awk -F' ' '{ printf \"%s\t%s\n\",\$1,\$3; }' | column -t | egrep ^/dev/ | sort"
alias gh='history|grep'

