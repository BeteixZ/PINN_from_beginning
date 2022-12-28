@echo off
::for %%i in (50,100,150,200,250) do (
::    for %%c in (3000,10000,17000,24000,31000,38000) do (
::        python main.py --layer 6 --neurons 60 --initpts %%i --bcpts %%i --colpts %%c --epochs 1200 --lr 1 --act mish --method lbfgs
::    )
::)
::for %%l in (2,4,6,8,10) do (
::    for %%n in (20,40,60,80,100,120) do (
::        python main.py --layer %%l --neurons %%n --initpts 200 --bcpts 200 --colpts 17000 --epochs 1200 --lr 1 --act mish --method lbfgs
::    )
::)

::for %%l in (0.02,0.1,1,2,5) do (
::    for %%e in (800,1000,1200,1400,1600) do (
::        python main.py --layer 6 --neurons 80 --initpts 200 --bcpts 200 --colpts 17000 --epochs %%e --lr %%l --act mish --method lbfgs
::    )
::)

for %%a in (tanh,gelu,elu,mish,softplus) do (
    python main.py --layer 6 --neurons 80 --initpts 200 --bcpts 200 --colpts 17000 --epochs 1200 --lr 1 --act %%a --method lbfgs
)

pause