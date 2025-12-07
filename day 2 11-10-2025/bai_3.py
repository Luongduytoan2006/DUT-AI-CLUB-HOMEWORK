def count_characters(word) -> dict:
    counts = {}
    for char in word:
        if char in counts:
            counts[char] += 1
        else:
            counts[char] = 1
    return counts

def count_word(paragraph) -> dict:
    counts = {}
    for word in paragraph.split():
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


if __name__ == "__main__":  
    word = input("Enter a word: ")
    result = count_characters(word)

    print(result)