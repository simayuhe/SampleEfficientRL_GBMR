

CUDA_VISIBLE_DEVICES=5 python mainGQv2.py --env="Pong-v4" --training_iters=10000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4, Bowling-v4, Frostbite-v4, Pong-v4,MontezumaRevenge-v4，

tmux a -t 10
CUDA_VISIBLE_DEVICES=0 python mainGQv2.py --env="Alien-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=0 python mainGQv2.py --env="Alien-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

tmux a -t 11
CUDA_VISIBLE_DEVICES=1 python mainGQv2.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=1 python mainGQv2.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

tmux a -t 12
CUDA_VISIBLE_DEVICES=2 python mainGQv2.py --env="Hero-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=2 python mainGQv2.py --env="Hero-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

tmux a -t 13
CUDA_VISIBLE_DEVICES=3 python mainGQv2.py --env="BankHeist-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=3 python mainGQv2.py --env="BankHeist-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

tmux a -t 14
CUDA_VISIBLE_DEVICES=4 python mainGQv2.py --env="Amidar-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=4 python mainGQv2.py --env="Amidar-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

tmux a -t 15
CUDA_VISIBLE_DEVICES=5 python mainGQv2.py --env="Bowling-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=5 python mainGQv2.py --env="Bowling-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.5 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

tumx a -t 16
CUDA_VISIBLE_DEVICES=6 python mainGQv2.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=6 python mainGQv2.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=1 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

tmux a -t 17
CUDA_VISIBLE_DEVICES=7 python mainGQv2.py --env="Pong-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=7 python mainGQv2.py --env="Pong-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410nq" --save_path="/home/kpl/SEGBMRresults/"

<!-- tmux a -t 18
CUDA_VISIBLE_DEVICES=1 python mainGQv2.py --env="MontezumaRevenge-v4" --training_iters=10000000 --memory_size=1000000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/" -->