import requests
from bs4 import BeautifulSoup

# Get the entry for the word "domāt"
response = requests.get('https://tezaurs.lv/api/retrieveEntry?hw=domāt')
print(response.text)
entry = response.text

# Parse the HTML to extract the meanings and usage examples
soup = BeautifulSoup(entry, 'html.parser')
meanings = [meaning.text for meaning in soup.find_all('span', class_='sv_NO')]
examples = [example.text for example in soup.find_all('span', class_='piem')]

# Print the results
print('Meanings:')
for meaning in meanings:
    print(f'- {meaning}')

print('\nUsage examples:')
for example in examples:
    print(f'- {example}')

