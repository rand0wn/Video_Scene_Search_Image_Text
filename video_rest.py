"""REST API for Video Training and Search"""

from flask import Flask, render_template
from flask import jsonify
import video_process

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# Train Video
@app.route("/train/<path:path>")
def train_video(path):
    path = "/" + str(path)[0:len(str(path)) - 1]
    # Train Videos
    video_obj = video_process.Video(path, 10, 0)
    return video_obj.train_videos()


# Search Similar Scene(Image)
@app.route("/isearch/<path:path>")
def search_image_scene(path):
    # Search From Image
    path = "/"+ str(path)[0:len(str(path))-1]
    return jsonify(video_process._ext_img_idx(path))


# Search Similar Scene(Text)
@app.route("/tsearch/<string:text>")
def search_text_scene(text):
    # Search From Text
    return jsonify(video_process._text_idx(text))

if __name__ == "__main__":
    app.run(host='localhost', port=5003)