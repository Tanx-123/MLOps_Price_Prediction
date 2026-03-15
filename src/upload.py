"""
Upload raw data CSV to S3 bucket.

Usage:
    python -m src.upload
    python -m src.upload --file data/raw/raw_data.csv
"""

import argparse
import logging
from src.core_utils import upload_to_s3, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Upload raw data to S3")
    parser.add_argument(
        "-f", "--file",
        default=config["data"]["raw_path"],
        help="Local file to upload",
    )
    parser.add_argument(
        "-b", "--bucket",
        default=config["s3"]["bucket"],
        help="S3 bucket name",
    )
    parser.add_argument(
        "-k", "--key",
        default=config["s3"]["raw_key"],
        help="S3 object key",
    )
    args = parser.parse_args()

    logger.info(f"Uploading raw data: {args.file} → s3://{args.bucket}/{args.key}")
    success = upload_to_s3(args.file, args.bucket, args.key)

    if success:
        logger.info("Raw data uploaded successfully!")
    else:
        logger.error("Failed to upload raw data.")

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
