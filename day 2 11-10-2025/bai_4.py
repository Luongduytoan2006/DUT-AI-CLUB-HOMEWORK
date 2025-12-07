from bai_3 import count_characters, count_word
import os

def read_input_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        content = file.read().strip()
    
    return content


if __name__ == "__main__":
    string_input = read_input_file("input.txt")
    print(f"Input string from file: {string_input}")
    result1, result2 = count_characters(string_input), count_word(string_input)
    
    print("Character counts:", result1)
    print("Word counts:", result2)

