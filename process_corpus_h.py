def process_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    sentences = []
    sentence = []
    labels = []
    label = []

    for line in content:
        line = line.strip()
        if len(line) <= 0:
            continue
        elif line == "dXYZ-":
            if sentence:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
        elif line[0] == "-":
            continue
        else:
            parts = line.split('\t')
            words = parts[0].lower().split(' ')
            if len(parts) > 2 and 's:' in parts[2]:
                sense_id = parts[2].split('s:')[1].split(' ')[0]
                label.extend([int(sense_id)] * len(words))
            else:
                label.extend([-100] * len(words))
            sentence.extend(words)

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels