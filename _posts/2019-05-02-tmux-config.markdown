---
layout: post
comments: true
title:  "Configuring tmux for more intuitive hot keys and better looking UI"
excerpt: ""
date:   2019-05-12 01:21:00
mathjax: true
---

<!--div class="imgcap">
<img src="/assets/Creating-your-own-dl-framework/framework.png" width="70%">

<div class="thecap">(Image credit: <a href="http://cs231n.github.io/neural-networks-3/">cs231n</a>).</div>
</div-->

Recently I came across this awesome tool called tmux. I use it as an substitute for VNC viewer. Unlike VNC viewer, it doesn't load the entire desktop but the terminal session. I like it because I don't always want the GUI of the server. In this post, we shall go through the installation process and make tmux hot keys more intuitive and usable. 

#### Installing TMUX on mac

If homebrew is not installed then install it: 

`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

Install tmux using the following command

`brew install tmux`

Confirm the installation

`tmux -V`

#### TMUX configuration 

By default, tmux hotkeys are very un-intuitive and hard to use. You can change the hot key mapping by creating and modifying a ~/.tmux.conf file. When I started to create my tmux config file I did not understand how to do it. Below, is my .tmux.conf file with description of each part (for a mac).  Hope this will help you create your own tmux configuration. 

```bash
# TMUX configuration file
# Unbind the stupid default cntrl-b prefix
unbind C-b 

# Make cntrl-space as the new prefix
# From now on: prefix == Cntrl-Space
# if you want to detach from a tmux session:
# prefix-d
set -g prefix C-Space
bind C-Space send-prefix

# Few settings to make tmux better
set -g default-terminal "screen-256color"
set -sg escape-time 0
set -g base-index 1
set -g bell-action any
setw -g pane-base-index 1
set -g history-limit 10000
setw -g monitor-activity on
set -g visual-activity off
set -g status-keys vi
setw -g mode-keys vi
setw -g xterm-keys on

#------------------------------------------------------------------------------
# Key Binding.
#------------------------------------------------------------------------------
# Easily load the source file (prefix-r)
bind r source-file ~/.tmux.conf

# Pane splitting.
# To make pane splitting more intuitive I mapped it to keys 
#	prefix \ for vertical 
# prefix - for horizontal
bind \ split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# Create new window at the current path
# prefix c
bind c new-window -c "#{pane_current_path}"

# Kill pane: prefix x
bind x kill-pane \; move-window -r\; setw automatic-rename

# Kill window: prefix &
bind & kill-window \; move-window -r\; setw automatic-rename

# We bind easy keys o and p to move between windows
# Cntrl-0 for previous window 
# Cntrl-p for next window 
bind -n C-o select-window -t -1
bind -n C-p select-window -t +1

# We bind easy arrow keys to move between panes
# Cntrl-Up_arrw for pane above
# Cntrl-down_arrw for pane below
# Cntrl-left_arrw for pane on left
# Cntrl-right_arrw for pane on right
bind -n C-Left select-pane -L
bind -n C-Right select-pane -R
bind -n C-Up select-pane -U
bind -n C-Down select-pane -D

# We bind easy arrow keys to resize panes
# Alt-Left/Right/Up/Down
bind -n M-Left resize-pane -L 5
bind -n M-Down resize-pane -D 5
bind -n M-Up resize-pane -U 5
bind -n M-Right resize-pane -R 5

# Do not display the original window's name when renaming it. This makes
# renaming faster since one does not need to first erase the original name.
bind , command-prompt -p "(rename-window '#W')" "rename-window '%%'"

# Make b start copy mode.
bind b copy-mode

# To start selection: Ctrl-Space + v
# To copy: y
# To paste in  vim : p
# To paste else where : Command + v
bind -t vi-copy v begin-selection
bind -t vi-copy C-v rectangle-toggle
bind -t vi-copy y copy-pipe "xclip -filter -selection clipboard | xclip -selection primary"

# TODO: Toggle between mouse mode UPDATE
# set -g mouse on
# bind -n C-m set-window-option mouse

#------------------------------------------------------------------------------
# Styling.
#------------------------------------------------------------------------------
# Status line.
set -g status-fg white
set -g status-bg black
set -g status-left ""
set -g status-right "#{?pane_synchronized, #[bg=blue]SYNCHRONIZED#[default],} #S "

# Window list.
setw -g window-status-fg colour246 # grey
setw -g window-status-bg black
setw -g window-status-attr none
setw -g window-status-format "#[fg=colour172]#I#[fg=white] #W "

# Active window.
setw -g window-status-current-fg white
setw -g window-status-current-bg black
setw -g window-status-current-attr bright
setw -g window-status-current-format "#[fg=colour172]#I#[fg=white] #W#F"

# Window activity.
setw -g window-status-activity-fg colour246 # grey
setw -g window-status-activity-bg black

# Panes.
set -g pane-border-fg white
set -g pane-border-bg black
set -g pane-active-border-fg red
set -g pane-active-border-bg black

# Command/message line.
set -g message-fg white
set -g message-bg black
set -g message-attr bright

# Status update interval.
set -g status-interval 60

# Make the window list appear at the left-hand side instead of at the center.
set -g status-justify left
```

#### Managing tmux sessions

I created a simple bash script and added it to ~/.bash_profile file to manage my tmux sessions. I know there are some plugins that do similar things, but I always find it easier to create a small script and know exactly what it does :-)

With this following script, 

- `tm` : opens the default session (if already created) or creates a new default session with name "0"
- `tm session_name`: opens session with name session_name (if already created) or creates a new session with name "session_name"

``` bash
# TMUX SESSION MANAGEMENT ~~~~
tm() {
        if [ $# -eq 0 ]
                then
                        echo "No Tmux session provided, loggin into the default tmux session."
                        #tmux attach -t $USER"_default" || tmux new -s $USER"_default"
                        tmux attach -t 0 || tmux new -s 0
        else
                echo "Attaching/creating session $1"
                tmux attach -t $1 || tmux new -s $1
        fi
}
```

You can find my complete tmux scripts at my [github](https://github.com/amohant4/tmux_conf). Happy tmuxing.

 

















### 



 