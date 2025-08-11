def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def chunk_text(text, max_chunk_size=500):
    """
    Split text into chunks of max_chunk_size words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i+max_chunk_size])
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    text = load_text("document2.txt")
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print("Sample chunk:")
    print(chunks[0])