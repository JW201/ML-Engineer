import argparse
import os
from datetime import datetime
import joblib
from sentence_transformers import SentenceTransformer

# Constants
MODEL_PATH = "/opt/huggingface_models/all-MiniLM-L6-v2"
SVM_MODEL_FILE = os.path.join(os.path.dirname(__file__), "model", "svm_model.pkl")

def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Score news headlines with sentiment classifier.")
    parser.add_argument("input_file", help="Text file with one headline per line.")
    parser.add_argument("source", help="Source name (e.g., nyt, chicagotribune).")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        parser.error(f"Input file '{args.input_file}' does not exist.")

    return args.input_file, args.source

def load_models():
    """Load the embedding and classification models."""
    print(" Loading SentenceTransformer model...")
    embedder = SentenceTransformer(MODEL_PATH)

    print(" Loading trained SVM classifier...")
    try:
        classifier = joblib.load(SVM_MODEL_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to load SVM model: {e}")
    
    return embedder, classifier

def read_headlines(file_path):
    """Read headlines from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        headlines = [line.strip() for line in f if line.strip()]
    print(f" Loaded {len(headlines)} headlines.")
    return headlines

def write_predictions(predictions, headlines, source):
    """Write predictions to output file."""
    today_str = datetime.today().strftime("%Y_%m_%d")
    output_file = f"headline_scores_{source}_{today_str}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for label, headline in zip(predictions, headlines):
            f.write(f"{label},{headline}\n")

    print(f" Results written to {output_file}")

def main():
    input_file, source = parse_arguments()
    embedder, classifier = load_models()
    headlines = read_headlines(input_file)

    print(" Embedding headlines...")
    embeddings = embedder.encode(headlines)

    print(" Predicting sentiment...")
    predictions = classifier.predict(embeddings)

    write_predictions(predictions, headlines, source)

if __name__ == "__main__":
    main()



