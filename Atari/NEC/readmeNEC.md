
执行：

CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Hero-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

这个记忆规模是50万 * num_action

Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4, Bowling-v4, Frostbite-v4, Pong-v4,MontezumaRevenge-v4，

1
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Alien-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

2
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

3
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Hero-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

4
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="BankHeist-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

5
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Amidar-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

6
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Bowling-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

7
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

8
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"

9
CUDA_VISIBLE_DEVICES=3 python mainNEC.py --env="MontezumaRevenge-v4" --training_iters=40000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0410" --save_path="/data/kyx_data/SEGBMRresults"