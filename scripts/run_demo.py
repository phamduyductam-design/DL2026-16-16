"""Web demo launcher owned by Người 6."""

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "deployment/app.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
