import os
import boto3
import logging
import argparse
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

S3_BUCKET = "price-trend-tanx"
S3_KEY = "raw_data/raw_data.csv"
LOCAL_PATH = "data/raw/raw_data.csv"

def fetch_from_s3(bucket, key, path, region='us-east-1'):
    """Download file from S3 and replace local file."""
    try:
        s3 = boto3.client('s3', region_name=region)
        logger.info(f"Downloading {key} from s3://{bucket}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file(bucket, key, path)
        logger.info(f"Downloaded to {path}")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials missing. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or run 'aws configure'")
        return False
    except PartialCredentialsError:
        logger.error("Incomplete AWS credentials")
        return False
    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'NoSuchBucket':
            logger.error(f"Bucket {bucket} not found")
        elif code == 'NoSuchKey':
            logger.error(f"Object {key} not found in {bucket}")
        elif code == 'AccessDenied':
            logger.error(f"Permission denied for bucket {bucket}. Ensure IAM user has s3:GetObject permission")
        else:
            logger.error(f"S3 error: {e}")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fetch data from S3')
    parser.add_argument('-b', '--bucket', default=S3_BUCKET, help='S3 bucket name')
    parser.add_argument('-k', '--key', default=S3_KEY, help='S3 object key')
    parser.add_argument('-f', '--file', default=LOCAL_PATH, help='Local file path')
    parser.add_argument('-r', '--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    success = fetch_from_s3(args.bucket, args.key, args.file, args.region)
    exit(0 if success else 1)

if __name__ == '__main__':
    main()
