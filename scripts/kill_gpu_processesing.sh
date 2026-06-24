#!/bin/bash

tail -f scripts/process_all.log | while read -r line; do
  if echo "$line" | grep -q '─── Run 2/10'; then
    echo "Run 2 starting — killing process_all"
    kill 135811
    kill 135839
    break
  fi
done &
echo "Watcher PID: $!"
