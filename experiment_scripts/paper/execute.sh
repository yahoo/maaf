incomplete=1
filename="execution.txt"

while [[ -s $filename ]]; do
  command=$(sed -n '$p' $filename)
  sed -i '$d' $filename
  echo $command
  bash $command
  sleep 1
done
