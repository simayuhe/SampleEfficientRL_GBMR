

8 :python
CUDA_VISIBLE_DEVICES=4 python mainGBMRv1.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410_1" --expert_memory_size=20 --save_path="/data/kyx_data/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=4 python mainGBMRv1.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410_1" --expert_memory_size=20 --save_path="/data/kyx_data/SEGBMRresults/"

Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4, MontezumaRevenge-v4，Bowling-v4, Frostbite-v4, Pong-v4
1 tmux a -t 21 这里只替换了一个add by trj的时候的更新方式
CUDA_VISIBLE_DEVICES=4 python mainGBMRv1.py --env="Alien-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/"


7python 中只改了动作执行
CUDA_VISIBLE_DEVICES=1 python mainGBMRv1_1.py --env="Alien-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_2" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/"

而且没有进入重构的过程，是不是那个参数调节过小的原因？

9PYTHON
CUDA_VISIBLE_DEVICES=7 python mainGBMRv1_2.py --env="Alien-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0410_3" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/"


tmux a -t 1
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

tmux a -t 2
CUDA_VISIBLE_DEVICES=2 python mainGBMRv2.py --env="Hero-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

tmux a -t 3
CUDA_VISIBLE_DEVICES=3 python mainGBMRv2.py --env="BankHeist-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

tmux a -t 4
CUDA_VISIBLE_DEVICES=4 python mainGBMRv2.py --env="Amidar-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

tmux a -t 5
CUDA_VISIBLE_DEVICES=5 python mainGBMRv2.py --env="Bowling-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

tmux a -t 6
CUDA_VISIBLE_DEVICES=6 python mainGBMRv2.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

tmux a -t 7
CUDA_VISIBLE_DEVICES=7 python mainGBMRv2.py --env="Pong-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

tmux a -t 8
CUDA_VISIBLE_DEVICES=0 python mainGBMRv2.py --env="MontezumaRevenge-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"