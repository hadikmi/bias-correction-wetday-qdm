import argparse
import yaml
from src.wetday_qdm import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
