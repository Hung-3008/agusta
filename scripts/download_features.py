"""
Download medarc/algonauts_2025.features from HuggingFace.
Run with: nohup python scripts/download_features.py > download_features.log 2>&1 &
Check progress: tail -f download_features.log
"""
import logging
import sys
from datetime import datetime

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REPO_ID = "medarc/algonauts_2025.features"
LOCAL_DIR = "/media/hung/HDD/workplaces/codes/multimodal/agusta/Data/algonauts_2025.features"


def main():
    logger.info(f"Starting download: {REPO_ID}")
    logger.info(f"Destination: {LOCAL_DIR}")
    logger.info(f"Start time: {datetime.now()}")

    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
        )
        logger.info(f"Download complete! Files saved to: {path}")
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
