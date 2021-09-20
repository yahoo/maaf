# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


incomplete=1
filename="execution.txt"

while [[ -s $filename ]]; do
  command=$(sed -n '$p' $filename)
  sed -i '$d' $filename
  echo $command
  bash $command
  sleep 1
done
