#for act in 'tanh' 'gelu' 'elu' 'leakyrelu' 'mish' 'softplus'
#do
#  python main.py --layer 6 --neurons 50 --initpts 50 --bcpts 50 --colpts 9000 --epochs 1000 --lr 1 --act $act
#done


#for epoch in {500..1500..250}
#do
#  for lr in 0.01 0.1 0.5 1 2 5
#  do
#    python main.py --layer 4 --neurons 40 --initpts 50 --bcpts 50 --colpts 5000 --epochs $epoch --lr $lr
#  done
#done

#for layer in {2..10..2}
#do
#  for neurons in {10..90..20}
#  do
#    python main.py --layer $layer --neurons $neurons --initpts 50 --bcpts 50 --colpts 5000 --epochs 1000 --lr 1
#  done
#done


for initpts in {50..250..50}
do
  for colpts in {3000..33000..6000}
  do
    python main.py --layer 6 --neurons 60 --initpts $initpts --bcpts $initpts --colpts $colpts --epochs 1000 --lr 1 --act 'mish' --method 'lbfgs'
  done
done