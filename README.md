# ML Trader

An automated trading bot that uses an ML model to infer the direction of stock/crypto prices


## Dev Setup
Setup Python env:
https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-macos


### Start python env
source .venv/bin/activate


pip install pip-tools

### create reqs file in requirements.in, use pip-tools to make minimal reqs.txt file
pip-compile requirements.in





### Install reqs
pip install -r requirements.txt



### (optional) Create bloated req file
pip freeze > requirements.txt





### server commands
#### start ssh agent
eval "$(ssh-agent -s)"

ssh-add ~/github

#### python
source ~/pyEnvs/ml-trader/bin/activate

pkill python
nohup python api.py &
ps aux

#### nodejs

forever start bin/www
forever list
forever stopall