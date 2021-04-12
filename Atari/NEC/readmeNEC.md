
执行：

CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Hero-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

这个记忆规模是50万 * num_action

Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4, Bowling-v4, Frostbite-v4, Pong-v4,MontezumaRevenge-v4，

1 已经有了
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Alien-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

2 已经有了
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

3 已经有了
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Hero-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

4 已经有了
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="BankHeist-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

5 已经有了
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Amidar-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

6  tmux a -t 19
CUDA_VISIBLE_DEVICES=2 python mainNEC.py --env="Bowling-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

7 已经有了
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

8 tmux a -t 20 
CUDA_VISIBLE_DEVICES=4 python mainNEC.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"

9
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="MontezumaRevenge-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/home/kpl/SEGBMRresults/"