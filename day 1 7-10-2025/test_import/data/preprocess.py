import data.loader as loader

def preprocess_data():
    loader.load_data()
    print("Data preprocessed successfully.")

if __name__ == "__main__":
    preprocess_data()