df -h /usr?            > /tmp/`date +%Y%m%d`-`hostname -s`.du
du -hs /usr?/data/*/* >> /tmp/`date +%Y%m%d`-`hostname -s`.du
ls /tmp/*.du | awk ' { printf ("%s %s\n", last, $1); last=$1 } ' | tail -1 | xargs diff > /tmp/`date +%Y%m%d`-`hostname -s`.delta
