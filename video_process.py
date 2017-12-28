"""Class for training video for search and performing other operations."""
import nltk
import numpy as np
import cv2
import pandas as pd
import run_inference
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error as mse
import random
import Config
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

# Video Data
vd_df = pd.read_csv(Config.vd_data+'/vd_data.csv')
len_vd_df = vd_df.shape[0]


class Video(object):

    def __init__(self,
                 filenames,
                 frame_frequency,
                 audio_or_sub):

        # Class Parameters
        self.filenames = filenames.split(',')   # Video Files
        self.frame_frequency = frame_frequency  # Number of frames to consider
        self.audio_or_sub = audio_or_sub        # With audio or subs

    # Train Videos and Add Data to DataFrame
    def train_videos(self):
        global vd_df
        global len_vd_df
        for filename in self.filenames:
            # init for video
            vd_cap = cv2.VideoCapture(filename)
            frame_count = 0
            video_name = filename.split('/')[len(filename.split('/'))-1]

            while vd_cap.isOpened():
                ret, frame = vd_cap.read()
                # Check for video
                if ret:
                    frame_count = frame_count + 1

                    # For every n secs, 24 fps
                    curr_frame = random.randint(1, self.frame_frequency * 24)  # Take a random frame between interval
                    if frame_count % curr_frame == 0:
                        frame_img_name = video_name + '_' + str(frame_count/24) + '.jpg'
                        frame_img_loc = Config.vd_data + '/frames/' + frame_img_name
                        cv2.imwrite(frame_img_loc, frame)

                        vd_df = vd_df.append({'frame': frame_img_name, 'video': video_name, 'time': frame_count/24, 'prob': 0, 'caps': 0, 'words': 0, 'tags': 0, 'subs': 0}, ignore_index=True)
                else:
                    # List of All Video Frames
                    frame_list = [Config.vd_data+'/frames/' + x for x in list(vd_df['frame'][len_vd_df:,])]

                    # Store Show and Tell Model Captions and Prob
                    file_input = [Config.model_checkpoint, Config.model_vocab, ",".join(frame_list)]
                    prob, cap = run_inference.img_captions(file_input)
                    vd_df.iloc[len_vd_df:, vd_df.columns.get_loc("prob")] = prob
                    vd_df.iloc[len_vd_df:, vd_df.columns.get_loc("caps")] = cap

                    for i in range(0, len(cap)):
                        words = nltk.re.sub("[^a-zA-Z]", " ", str(cap[i]))
                        words = list(set(words.split(' ')))
                        stop = set(stopwords.words("english"))
                        rem_words = [w for w in words if not w in stop and len(w) > 2]
                        vd_df.iloc[len_vd_df+i, vd_df.columns.get_loc("words")] = str(rem_words)

                    # Update Final Changes to CSV
                    vd_df.to_csv(Config.vd_data + '/vd_data.csv', index=False)
                    len_vd_df = vd_df.shape[0]  # Update Length
                    break
        return "Training Completed"


# Image Frame Prob
def _map_frame_prob(str_frame_prob):
    return map(float, str_frame_prob[1:len(str_frame_prob) - 1].split(', '))


# Frame Prob Indexing
def _frame_indexing(image_idx, prob, name):
    image_matches = {}
    image_match = None

    if image_idx == -1:
        frame_prob = prob
        image_match = image_matches[name] = {}
    else:
        frame_prob = _map_frame_prob(vd_df['prob'][image_idx])
        image_match = image_matches[vd_df['frame'][image_idx]] = {}

    for i in range(0, len(vd_df) - 1):
        mse_prob = mse(frame_prob, _map_frame_prob(vd_df['prob'][i + 1]))
        image_match[mse_prob] = {}
        image_match[mse_prob]['img'] = vd_df['frame'][i + 1]

    return image_matches


# External Image Indexing
def _ext_img_idx(path):
    file_input = [Config.model_checkpoint, Config.model_vocab, path]
    prob, cap = run_inference.img_captions(file_input)
    img_prob = _map_frame_prob(prob[0])
    print cap
    return _frame_indexing(-1, img_prob, path.split('/')[len(path.split('/'))-1])


# Sentence and Word Similarity
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    try:
        score /= count
    except:
        pass
    return score


# Text list to sent
def _map_sent(str_sent):
    if '[' in str_sent:
        return nltk.re.sub("[^a-zA-Z]", " ", str_sent[1:len(str_sent) - 1])
    else:
        return nltk.re.sub("[^a-zA-Z]", " ", str_sent)


# Text Indexing
def _text_idx(text):
    sent_match = {}
    for i in range(0, len(vd_df) - 1):
        sent_prob = sentence_similarity(_map_sent(text), _map_sent(vd_df['words'][i]))
        sent_match[1 - sent_prob] = {}
        sent_match[1 - sent_prob]['img'] = vd_df['frame'][i]
    return sent_match
