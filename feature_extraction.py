from img2vec_pytorch import Img2Vec
from pathlib import Path
from PIL import Image
import os, pickle


def extract_raw_images_features():

    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    img2vec = Img2Vec()

    raw_data = Path('dataset/raw')

    features = []
    labels = []

    for category in raw_data.iterdir():
        for img_path in category.iterdir():
            try:
                img  = Image.open(img_path)
                img_feature = img2vec.get_vec(img)
                features.append(img_feature)
                labels.append(category)
            except:
                pass

    result = {
            'features': features,
            'labels': labels
        }
    with open('raw_data_features.pkl', 'wb') as f:
        pickle.dump(result, f)

