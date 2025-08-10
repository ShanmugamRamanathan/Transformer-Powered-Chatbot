from transformers import pipeline

def main():
    # Load the pretrained Q&A pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Example context (can be replaced with any text)
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    """

    print("Welcome to the Smart Q&A Chatbot!")
    print("Ask a question about the context below:\n")
    print(context)
    print("\nType 'exit' to quit.")

    while True:
        question = input("\nYour question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        result = qa_pipeline(question=question, context=context)
        print(f"Answer: {result['answer']} (Confidence: {result['score']:.2f})")

if __name__ == "__main__":
    main()