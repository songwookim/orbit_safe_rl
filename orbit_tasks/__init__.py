"""Package containing task implementations for various robotic environments."""

import os
import toml
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import orbit_tasks.cartpole
import orbit_tasks.reach.config.ur_10