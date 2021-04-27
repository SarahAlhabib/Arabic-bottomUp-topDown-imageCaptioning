from dataset.arabic_dataset import create_input_files


if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files("Flickr8k_text/Flickr8k.arabic.full.txt")
