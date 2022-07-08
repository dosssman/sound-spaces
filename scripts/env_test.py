import argparse

from habitat.datasets import make_dataset
from ss_baselines.av_nav.config import get_config
from ss_baselines.common.environments import AudioNavRLEnv

config = get_config(
    config_paths="ss_baselines/av_nav/config/audionav/mp3d/interactive_demo.yaml",
    opts=None,
    run_type="eval")
config.defrost()
config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
config.freeze()
print(config)

dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
env = AudioNavRLEnv(config=config, dataset=dataset)

observations = env.reset()

print(observations)