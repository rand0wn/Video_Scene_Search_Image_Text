import run_inference


if __name__ == "__main__":
    # File Inputs
    file_input = ['models/model2.ckpt-2000000', 'models/word_counts.txt', 'imgs/army1.jpg']
    probs, captions = run_inference.img_captions(file_input)
    print(probs)
    for i in captions: print(i)