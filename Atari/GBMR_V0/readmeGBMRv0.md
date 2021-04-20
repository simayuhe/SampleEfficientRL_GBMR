
日期 2021年4月20日 
ip : gbmr

tmux a -t 0
CUDA_VISIBLE_DEVICES=0 python mainGBMRv0.py --env="Alien-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"

tmux a -t 1
CUDA_VISIBLE_DEVICES=1 python mainGBMRv0.py --env="MsPacman-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"

tmux a -t 2
CUDA_VISIBLE_DEVICES=2 python mainGBMRv0.py --env="Amidar-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"

tmux a -t 3
CUDA_VISIBLE_DEVICES=3 python mainGBMRv0.py --env="Hero-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"

tmux a -t 4
CUDA_VISIBLE_DEVICES=4 python mainGBMRv0.py --env="BankHeist-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.04 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"

tmux a -t 5
CUDA_VISIBLE_DEVICES=5 python mainGBMRv0.py --env="Pong-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.04 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"

tmux a -t 6 
CUDA_VISIBLE_DEVICES=0 python mainGBMRv0.py --env="Bowling-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=0.04 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"

tmux a -t 7
CUDA_VISIBLE_DEVICES=1 python mainGBMRv0.py --env="Frostbite-v4" --training_iters=40000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0419" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRV0"
