import re

def remove_broken_hyphens(text):
    # Join words broken with hyphen + newline
    text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)
    # Replace remaining single newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    cleaned_text = remove_broken_hyphens(raw_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"Cleaned text written to {output_path}")


# Example usage
if __name__ == "__main__":
    input_filename = "data/panchatantra-ryder.txt"   # change to your input file
    output_filename = "panchatantra-ryder-clean.txt"
    clean_file(input_filename, output_filename)