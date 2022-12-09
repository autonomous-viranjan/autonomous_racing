# autonomous_racing
- ROS package for autonomous racing using optimal control
- Contains ROS nodes to run high level planner and low level tracking controller together

## Description:
### High level:
- `pdgplanner.py` is the high-level planner of the Ego vehicle
- `aiagent.py` contains competing AI agents

### Low Level:
- `tracking_MPC_allConstr.mexa64` is the the controller compiled from ACADO Toolkit
- `nlmpc_autorace_sim_allConstr.m` is the low level control simulation ros node

## Key dependencies:
- `pdgplanner.py` should be in the same directory as `ilqr/` folder
- `aiagent.py` requires Casadi in the appended path
- Python 2.7
- Numpy 
- Theanos
- ACADO toolkit for MATLAB (https://acado.github.io/matlab_overview.html)
- ROS Toolbox from MATLAB

## Contributors:
- Viranjan Bhattcharyya
- Jacky Tang
- Prakhar Gupta

## Instructions to run:
### Clone the project:
```bash
$ cd ~/catkin_ws/src
$ git clone https://github.com/pgupta2050/autonomous_racing.git
$ roscore
$ rosrun autonomous_racing high_level_node.py
```

### To run the low levl node:
Run the Matlab file [nlmpc_autorace_sim_allConstr](https://github.com/pgupta2050/autonomous_racing/blob/main/src/low_level/Prakhar/nlmpc_autorace_sim_allConstr.m)

