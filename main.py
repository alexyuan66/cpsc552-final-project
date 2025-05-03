from model.open_coder import OpenCoder


if __name__ == "__main__":
    # Initialize model
    model = ...

    # Pipeline should be the model's prompting function that takes in a prompt (str) and outputs a response (str)
    pipeline = model.pipeline

    # Initialize OpenCoder framework that wraps around the model's prompting function
    openCoder = OpenCoder(pipeline)

    # Example of generating a response using OpenCoder
    query = 'How do I fix an index out of range error in Python?'
    print(openCoder.generate(query))
