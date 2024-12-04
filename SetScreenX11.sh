# Scanning screen setup
# DP-6 = BOLD screen
# DP-2 = inverted iiyama screen
# DP-1 = iiyama screen
xrandr --output DP-1 --primary --mode 1920x1080 --rate 120
#xrandr --output DP-6 --primary
xrandr --output DP-2 --mode 1920x1080 --rate 120 --rotate inverted
xrandr --output DP-6 --mode 1920x1080 --rate 120 --rotate inverted --right-of DP-1 --output DP-2 --same-as DP-6
#xrandr --output DP-2 --mode 1920x1080 --rate 120
#xrandr --output DP-6 --right-of DP-1
#xrandr --output DP-2 --same-as DP-6
