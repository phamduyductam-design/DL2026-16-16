import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "configs" / "data.yaml",
    ROOT / "docs" / "PROTOCOL.md",
    ROOT / "docs" / "WORKFLOW.md",
    ROOT / "docs" / "TASKS.md",
    ROOT / "data" / "README.md",
    ROOT / "src" / "README.md",
    ROOT / "outputs" / "README.md",
]


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        print("Missing required project paths:")
        for path in missing:
            print(f"- {path}")
        return 1

    forbidden = list((ROOT / "data").glob("MVTecAD/**/*"))
    if forbidden:
        print("Dataset files were found inside the repository. Keep MVTec AD outside Git.")
        return 1

    print("Project structure is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

