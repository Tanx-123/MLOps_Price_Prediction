"""
Upload raw data CSV to S3.

Usage:
    python -m src.upload
    python -m src.upload --file data/raw/raw_data.csv
"""
import sys
import argparse
import logging

from src.core_utils import upload_to_s3, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    s3 = config["s3"]

    parser = argparse.ArgumentParser(description="Upload raw data to S3")
    parser.add_argument("-f", "--file", default=config["data"]["raw_path"])
    parser.add_argument("-b", "--bucket", default=s3["bucket"])
    parser.add_argument("-k", "--key", default=s3["raw_key"])
    args = parser.parse_args()

    logger.info(f"Uploading {args.file} → s3://{args.bucket}/{args.key}")
    success = upload_to_s3(args.file, args.bucket, args.key)

    if success:
        logger.info("Done!")
    else:
        logger.error("Upload failed.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
