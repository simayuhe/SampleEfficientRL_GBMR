

8 :python
CUDA_VISIBLE_DEVICES=4 python mainGBMRv1.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410_1" --expert_memory_size=20 --save_path="/data/kyx_data/SEGBMRresults/"

CUDA_VISIBLE_DEVICES=4 python mainGBMRv1.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.1 --riqi="0410_1" --expert_memory_size=20 --save_path="/data/kyx_data/SEGBMRresults/"

Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4, MontezumaRevenge-v4，Bowling-v4, Frostbite-v4, Pong-v4