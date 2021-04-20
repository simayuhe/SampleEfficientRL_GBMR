执行：

CUDA_VISIBLE_DEVICES=3 python mainGQv1.py --env="Hero-v4" --training_iters=10000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

CUDA_VISIBLE_DEVICES=3 python mainGQv1.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

Alien-v4 

1:PYTHON

CUDA_VISIBLE_DEVICES=3 python mainGQv1.py --env="Alien-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"\

2:PYTHON
CUDA_VISIBLE_DEVICES=3 python mainGQv1.py --env="Alien-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

MsPacman-v4
3:PYTHON
CUDA_VISIBLE_DEVICES=2 python mainGQv1.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"
4:PYTHON
CUDA_VISIBLE_DEVICES=2 python mainGQv1.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"


Frostbite
5PYTHON
CUDA_VISIBLE_DEVICES=4 python mainGQv1.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

5PYTHON
CUDA_VISIBLE_DEVICES=4 python mainGQv1.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_2" --save_path="/data/kyx_data/SEGBMRresults"

Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4, Bowling-v4, Frostbite-v4, Pong-v4,MontezumaRevenge-v4，

1
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="Alien-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

2
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

3
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="Hero-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

4
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="BankHeist-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

5
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="Amidar-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

6
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="Bowling-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

7
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

8
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

9
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="MontezumaRevenge-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults/"

日期 2021年4月20日
IP： gbmr

tmux a -t 8
CUDA_VISIBLE_DEVICES=2 python mainGQv1.py --env="Alien-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"

tmux a -t 9
CUDA_VISIBLE_DEVICES=3 python mainGQv1.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"

tmux a -t 10
CUDA_VISIBLE_DEVICES=4 python mainGQv1.py --env="Hero-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"

tmux a -t 11
CUDA_VISIBLE_DEVICES=5 python mainGQv1.py --env="BankHeist-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.04 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"

tmux a -t 12
CUDA_VISIBLE_DEVICES=0 python mainGQv1.py --env="Amidar-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"

tmux a -t 13
CUDA_VISIBLE_DEVICES=1 python mainGQv1.py --env="Bowling-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.04 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"

tmux a -t 14
CUDA_VISIBLE_DEVICES=2 python mainGQv1.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"

tmux a -t 15
CUDA_VISIBLE_DEVICES=3 python mainGQv1.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.04 --riqi="0419" --save_path="/home/kpl/SEGBMRresults/GQV1"