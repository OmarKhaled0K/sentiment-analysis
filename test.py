from helper import *

def main():
    # Load the model
    classifier = load_model()

    while True:
        text = input("Enter Your review: ")
        # press q to exit
        if text == 'q':
            break
        sentiment, confidence = predict_sentiment_aspect(text, classifier)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence}")
        print("---")


if __name__ == "__main__":
    main()