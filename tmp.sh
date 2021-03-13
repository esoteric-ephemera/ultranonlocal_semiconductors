a0=5.43

for delta in 0.1 0.05 0.02 0.01 0.005
do
  for step in 0.0 -1.0 1.0 -2.0 2.0 -3.0 3.0 -4.0 4.0 -5.0 5.0
  do
  ta0=$(echo "${a0}+$step*$delta" | bc)
  echo $ta0
  done
done
