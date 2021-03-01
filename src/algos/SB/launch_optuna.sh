#!/bin/bash
#

num_instances="$1"
script_name="$2"
session="$3"

# set up tmux
tmux start-server

# create a new tmux session
tmux new-session -d -s $session

tmux send-keys "python ${script_name}" C-m;
for ((n=0; n < (num_instances-1); n++));
do
  sleep 7;
	tmux splitw -v;
	tmux send-keys "python ${script_name}" C-m;
done

#tmux attach-session -t $session
