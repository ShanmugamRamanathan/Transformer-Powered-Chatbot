from transformers import pipeline
from search_chunks import load_index, search

def main():
    # Load FAISS index and document chunks
    index, chunks = load_index()

    # Load Q&A model
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")

    print("Smart Q&A Chatbot Ready! Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Search for top relevant chunks
        top_chunks = search(question, index, chunks, top_k=3)

        answers = []
        # Ask Q&A model on each chunk
        for chunk in top_chunks:
            result = qa_pipeline(question=question, context=chunk)
            answers.append(result)

        # Pick the answer with highest confidence
        best_answer = max(answers, key=lambda x: x['score'])
        print(f"\nAnswer: {best_answer['answer']} (Confidence: {best_answer['score']:.2f})")

if __name__ == "__main__":
    main()