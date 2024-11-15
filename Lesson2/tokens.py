



import spacy
import spacy
#python -m spacy download en_core_web_sm

def token():
    nlp = spacy.load('en_core_web_sm')  # Load the English tokenizer
    doc = nlp("This is a sentence that we're going to tokenize.")
    tokens = [token.text for token in doc]
    print(tokens)


    nlp = spacy.load('en_core_web_sm')
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)



def main():
    print("Hello, world!")
    token()

if __name__ == "__main__":
    # Executes the main function only if this file is executed as the main script (not imported as a module)
    main()