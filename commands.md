
```sh
# activate ROS environment
cd Desktop/lerobot_ws
conda deactivate 
source install/setup.bash
ros2 launch teleop_leaders leader_hw_single_omy.launch.py 

# activate lerobot env
conda activate lerobot
cd Desktop/lerobot-mujoco-tutorial
# run sim with omy arm controller
python3 create_dataset_parple.py -f papras_7dof -l puppeteer_omy 
```