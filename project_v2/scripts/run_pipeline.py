import yaml
from pathlib import Path

from processing.pipeline import Pipeline

def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pipe = Pipeline(config)
    result = pipe.run()

    print("Pipeline complete.")
    print(result.keys())

if __name__ == "__main__":
    main()