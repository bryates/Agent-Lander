# Using Deep Q-learning and reinforcement learning to train AI agents
## Skills
ML: AI Agent Deep Q-Network (DQN), Unsupervised Learning, Reinforcement Learning<br>
Python: PyTorch, numpy, OpenAI Gymnasium

## Executive Summary
Training ML models can be a time demanding task.
Instead of programming a set of rules or paths a model should follow, we can use **unsupervised learning** to allow an AI agent to explore the space on its own, similar to how humans learn new tasks.
A simple set of rules rewards the agent for making good decisions (like successfully landing the spacecraft), and penalizes poor or wasteful decisions (like crashing or wasting fuel).
I used the concept of **reinforcement learning** with the OpenAI Gymnasium to teach a **DQN** to play Lunar Lander.

## Business Problem:  
How can reinforcement learning be applied to optimize autonomous decision-making in real-world systems like drones or robotic delivery?

## Methodology
1. Developed an **AI agent** with a DQN using **PyTorch**
1. Trained the agent on the Lunar Lander module in the **OpenAI Gymnasium** for several thousand episodes
1. Tuned hyper-parameters to find an optimal sate for efficient learning and exploration

## Results & Business Recommendations
A simple AI Agent using unsupervised learning is a powerful tool for training robots for autonomous decision making.
To mitigate earl expenses and reduce damage from sub-optimal environment exploration, we can start with software simulations of the environment and the sensors on our robot.
This will allow us to build a base model which we can deploy to physical robots.

## Next Steps
1. Extend the agent to more complex environments (e.g., 3D simulations or real-world robotics)
1. Explore cost effectiveness of software simulation
1. Design and implement software simulation to avoid basic obstacles
1. Deploy simple obstacle-avoidance models on physical robots
1. Allow robots to explore the environment to continue trailing/optimization

Example YouTube video:

[![Lander video on YouTube](https://img.youtube.com/vi/7CxRNg_K8JM/0.jpg)](https://youtu.be/7CxRNg_K8JM)


<br><br><br>

[![Python application](https://github.com/bryates/Agent-Lander/actions/workflows/python-os.yml/badge.svg)](https://github.com/bryates/Agent-Lander/actions/workflows/python-os.yml)

# Agent-Lander
Unsupervised reinforcement learning project
