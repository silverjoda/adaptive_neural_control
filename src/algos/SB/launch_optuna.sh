#!/bin/bash
#
session="$3"

# set up tmux
tmux start-server

# create a new tmux session
tmux new-session -d -s $session

x="$1"
script_name="$2"

tmux send-keys "python ${script_name}" C-m;	
for ((n=0; n < (x-1); n++));
do
  sleep 7;
	tmux splitw -v;
	tmux send-keys "python ${script_name}" C-m;	
done

#tmux attach-session -t $session
