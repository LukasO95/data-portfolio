# scripts/make_processed_data.py
import argparse
from src.data import make_processed_sample_listings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=3000)
    args = parser.parse_args()

    make_processed_sample_listings(sample_size=args.sample_size)

if __name__ == "__main__":
    main()