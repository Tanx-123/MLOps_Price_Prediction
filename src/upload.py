import argparse
import os
import boto3
import logging
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_file_to_s3(file_path, bucket, key, region='us-east-1'):
    """Upload a file to S3 with comprehensive error handling."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    try:
        s3 = boto3.client('s3', region_name=region)
        logger.info(f"Uploading {file_path} to s3://{bucket}/{key}")
        s3.upload_file(file_path, bucket, key)
        logger.info("Upload successful!")
        return True
        
    except NoCredentialsError:
        logger.error("AWS credentials missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or run 'aws configure'")
        return False
        
    except PartialCredentialsError:
        logger.error("Incomplete AWS credentials")
        return False
        
    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'AccessDenied':
            logger.error(f"Permission denied for bucket {bucket}. Ensure IAM user has s3:PutObject permission")
        elif code == 'NoSuchBucket':
            logger.error(f"Bucket {bucket} does not exist")
        else:
            logger.error(f"S3 error: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload file to S3')
    parser.add_argument('-f', '--file', default='data/raw/raw_data.csv', help='File to upload')
    parser.add_argument('-b', '--bucket', default='price-trend-tanx', help='S3 bucket name')
    parser.add_argument('-k', '--key', default='raw_data/raw_data.csv', help='S3 object key')
    parser.add_argument('-r', '--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    success = upload_file_to_s3(args.file, args.bucket, args.key, args.region)
    exit(0 if success else 1)

if __name__ == '__main__':
    main()