"""
Data validation utilities using Great Expectations style validation.

Usage:
    python -m src.validate_data
    python -m src.validate_data --stage clean
"""
import os
import sys
import argparse
import logging
from typing import Dict, List, Any

import pandas as pd
from src.core_utils import load_config, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of data validation."""
    
    def __init__(self):
        self.checks: List[Dict[str, Any]] = []
        self.passed: bool = True
        self.errors: List[str] = []
    
    def add_check(self, name: str, passed: bool, message: str = ""):
        self.checks.append({
            "name": name,
            "passed": passed,
            "message": message,
        })
        if not passed:
            self.passed = False
            self.errors.append(f"{name}: {message}")
    
    def summary(self) -> Dict:
        return {
            "passed": self.passed,
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for c in self.checks if c["passed"]),
            "failed_checks": sum(1 for c in self.checks if not c["passed"]),
            "checks": self.checks,
            "errors": self.errors,
        }


class DataValidator:
    """Data validation for rental price prediction pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.features = config.get("features", {})
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that dataframe has expected columns."""
        result = ValidationResult()
        
        required_cols = (
            self.features.get("numerical", []) +
            self.features.get("categorical", []) +
            self.features.get("high_cardinality", []) +
            [self.features.get("target", "Rent")]
        )
        
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            result.add_check("schema", False, f"Missing columns: {missing}")
        else:
            result.add_check("schema", True, "All required columns present")
        
        return result
    
    def validate_nulls(self, df: pd.DataFrame) -> ValidationResult:
        """Check for null values."""
        result = ValidationResult()
        
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        
        if len(cols_with_nulls) > 0:
            result.add_check(
                "nulls", False, 
                f"Nulls found: {dict(cols_with_nulls)}"
            )
        else:
            result.add_check("nulls", True, "No null values")
        
        return result
    
    def validate_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Validate numerical columns are in reasonable ranges."""
        result = ValidationResult()
        
        # BHK should be 1-10
        if "BHK" in df.columns:
            invalid = (df["BHK"] < 1) | (df["BHK"] > 10)
            count = invalid.sum()
            if count > 0:
                result.add_check("BHK_range", False, f"{count} values outside 1-10")
            else:
                result.add_check("BHK_range", True, "BHK in valid range")
        
        # Size should be > 0 and < 10000
        if "Size" in df.columns:
            invalid = (df["Size"] <= 0) | (df["Size"] > 10000)
            count = invalid.sum()
            if count > 0:
                result.add_check("Size_range", False, f"{count} values <= 0 or > 10000")
            else:
                result.add_check("Size_range", True, "Size in valid range")
        
        # Rent should be > 0
        target = self.features.get("target", "Rent")
        if target in df.columns:
            invalid = df[target] <= 0
            count = invalid.sum()
            if count > 0:
                result.add_check("Rent_range", False, f"{count} non-positive rent values")
            else:
                result.add_check("Rent_range", True, "Rent in valid range")
        
        return result
    
    def validate_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """Check for duplicate rows (warning level check)."""
        result = ValidationResult()
        
        dup_count = df.duplicated().sum()
        # Duplicate check is a warning, not a failure
        if dup_count > 0:
            result.add_check(
                "duplicates", True,  # Pass but warn
                f"{dup_count} duplicate rows (warning only)"
            )
        else:
            result.add_check("duplicates", True, "No duplicates")
        
        return result
    
    def validate_cardinality(self, df: pd.DataFrame) -> ValidationResult:
        """Check cardinality of categorical columns."""
        result = ValidationResult()
        
        high_card = self.features.get("high_cardinality", [])
        for col in high_card:
            if col in df.columns:
                unique_count = df[col].nunique()
                # Warn if too few unique values (might indicate data issue)
                if unique_count < 5:
                    result.add_check(
                        f"cardinality_{col}", False,
                        f"Only {unique_count} unique values"
                    )
                else:
                    result.add_check(
                        f"cardinality_{col}", True,
                        f"{unique_count} unique values"
                    )
        
        return result
    
    def validate_all(self, df: pd.DataFrame, stage: str = "raw") -> ValidationResult:
        """Run all validation checks."""
        result = ValidationResult()
        
        # Schema check (always run first)
        schema_result = self.validate_schema(df)
        result.checks.extend(schema_result.checks)
        result.errors.extend(schema_result.errors)
        
        # Skip other checks if schema fails
        if not schema_result.passed:
            result.passed = False
            return result
        
        # Null check
        null_result = self.validate_nulls(df)
        result.checks.extend(null_result.checks)
        
        # Range check (skip for raw data with missing values)
        if stage != "raw":
            range_result = self.validate_ranges(df)
            result.checks.extend(range_result.checks)
        
        # Duplicate check
        dup_result = self.validate_duplicates(df)
        result.checks.extend(dup_result.checks)
        
        # Cardinality check
        card_result = self.validate_cardinality(df)
        result.checks.extend(card_result.checks)
        
        result.passed = all(c["passed"] for c in result.checks)
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Validate data")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--stage", 
        choices=["raw", "clean", "features"], 
        default="clean",
        help="Data stage to validate"
    )
    parser.add_argument(
        "--output", 
        default="artifacts/validation_results.json",
        help="Output file for validation results"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_path = config["data"]["processed_path"]
    
    validator = DataValidator(config)
    
    if args.stage == "raw":
        df = pd.read_csv(config["data"]["raw_path"])
    else:
        clean_path = os.path.join(data_path, "cleaned_data.csv")
        if not os.path.exists(clean_path):
            logger.error(f"Cleaned data not found: {clean_path}")
            sys.exit(1)
        df = pd.read_csv(clean_path)
    
    logger.info(f"Validating {args.stage} data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    result = validator.validate_all(df, stage=args.stage)
    summary = result.summary()
    
    logger.info("=" * 50)
    logger.info(f"VALIDATION RESULT: {'PASSED' if summary['passed'] else 'FAILED'}")
    logger.info(f"Checks: {summary['passed_checks']}/{summary['total_checks']} passed")
    
    if not summary['passed']:
        logger.error("Failures:")
        for error in summary['errors']:
            logger.error(f"  - {error}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_json(summary, args.output)
    logger.info(f"Results saved to {args.output}")
    
    sys.exit(0 if summary['passed'] else 1)


if __name__ == "__main__":
    main()
