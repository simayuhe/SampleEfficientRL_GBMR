CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410_1" --expert_memory_size=20 --save_path="/data/kyx_data/SEGBMRresults/" --KF="CrossNodes"

CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410_1" --expert_memory_size=20 --save_path="/data/kyx_data/SEGBMRresults/" --KF="CrossNodes"

Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4，Bowling-v4, Frostbite-v4, Pong-v4, MontezumaRevenge-v4

1
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="Alien-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

2
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

3
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="Hero-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

4
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="BankHeist-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

5
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="Amidar-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

6
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="Bowling-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

7
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

8
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="Pong-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"

9
CUDA_VISIBLE_DEVICES=1 python mainGBMRv2.py --env="MontezumaRevenge-v4" --training_iters=40000000 --memory_size=1000000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410_1" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/" --KF="CrossNodes"