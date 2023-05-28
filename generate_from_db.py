import psycopg2
import spacy_udpipe
import nltk
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizerFast


# Initialize the arrays to store the data
def generate_data():
    tokenizer = get_tokenizer('basic_english')  # Replace 'basic_english' with a Latvian tokenizer if available
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    # Connect to the PostgreSQL database
    connection = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="rrr22678max123",
        host="localhost",  # replace with your host if different
        port="5432"  # replace with your port if different
    )

    # Create a cursor to execute SQL queries
    cursor = connection.cursor()

    # Load the Latvian UDPipe model for lemmatization
    spacy_udpipe.download("lv")
    nlp = spacy_udpipe.load("lv")

    # Execute the SQL query to retrieve the required data
    query = """
        SELECT a.heading, b.id AS sense_id, b.gloss, c.content
        FROM dict.entries AS a
        JOIN dict.senses AS b ON a.id = b.entry_id
        JOIN dict.examples AS c ON b.id = c.sense_id;
    """
    cursor.execute(query)

    text_sentences = []
    replaced_sentences = []
    data = cursor.fetchall()
    # Iterate over the data and generate the training examples
    for row in data:
        heading, sense_id, gloss, content = row
        # Preprocess the data if needed (e.g., handle modifications)
        # Lowercase the content
        content = content.lower()
        # Process the content using the Latvian UDPipe model
        doc = nlp(content)

        replaced_words = []
        initial_tokens = []
        for token in doc:
            # Normalize the token using lemmatization
            normalized_token = token.lemma_
            # Replace the word with its appropriate sense if it matches the heading
            if normalized_token == heading.lower():
                replaced_word = sense_id
            else:
                # Execute the SQL query to retrieve the required data
                query = """
                    SELECT b.id, b.gloss AS sense
                    FROM dict.entries a
                    JOIN dict.senses b ON a.id = b.entry_id
                    WHERE a.heading = %s
                    ORDER BY b.id;
                """
                cursor.execute(query, (normalized_token,))
                repl = cursor.fetchone()
                if repl:
                    sense_id = repl[0]
                    replaced_word = sense_id
                else:
                    # print(normalized_token)
                    replaced_word = bert_tokenizer(str(token))['input_ids'][1]  # Assign a generic ID for words without a sense
            initial_tokens.append(str(token))
            replaced_words.append(replaced_word)

        # Join the replaced words to form the replaced sentence
        # replaced_sentence = ' '.join(replaced_words)

        # Add the original and replaced sentences to the arrays
        text_sentences.append(content)
        replaced_sentences.append(replaced_words)
    return text_sentences, replaced_sentences


#text, sense = generate_data()
# text_sentences, replaced_sentences = generate_data()

'''
# Print the data side by side
for i in range(len(text_sentences)):
    print("Original Sentence:", text_sentences[i])
    print("Replaced Sentence:", replaced_sentences[i])
    print()
'''
