import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from torchnmt import get_trainer
from torchnmt.config import Config


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    config = Config(**OmegaConf.to_container(cfg))
    trainer = get_trainer(config)
    trainer.train()


if __name__ == "__main__":
    train()
