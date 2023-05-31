import os
import random
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re  ###

keras_layer = hub.KerasLayer(
    'https://kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/152-feature-vector/versions/2')


main_folder = "venv\cropped_images"


folders = ["chris_evans", "chris_hemsworth", "mark_ruffalo", "robert_downey_jr", "scarlett_johansson"]


gallery_size = 100
img_per_folder = gallery_size // len(folders)
query_size = 5
top = 5
if top > len(folders):
    raise Exception("Think twice!")

query_images = []
sub_query_dir = []


gallery_images = []
sub_gallery_dir = []




def preprocess_image(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image


def feat_ext(query_list: list):
    res = []
    
    for el in query_list:
        query_features = keras_layer(tf.expand_dims(el, axis=0))
        res.append(query_features)
    
    return res


for folder in folders:
    folder_path = os.path.join(main_folder, folder)
    images = os.listdir(folder_path)
    random.shuffle(images) 
   
    for img in images[:img_per_folder]:
        sub_gal = re.match(r'^[A-Za-z_]+(?=\d)', images[:img_per_folder][0]).group()
        sub_gallery_dir.append(sub_gal)
    gallery_images.extend(images[:img_per_folder]) 
    
print('lenght of the gallery', len(gallery_images))

print('lenght of the sub_gallery', len(sub_gallery_dir))


gallery = []

i = 0
for image_file in gallery_images:
    image_path = os.path.join(main_folder, sub_gallery_dir[i], image_file)
    i += 1
    image = tf.io.read_file(image_path)
    image = preprocess_image(image)
    gallery.append(image)
i = 0


gallery_features = feat_ext(gallery)
print('lenght:', len(gallery_features))


query_images = []
sub_query_dir = []


all_images = []
for folder in folders:
    folder_path = os.path.join(main_folder, folder)
    images = os.listdir(folder_path)
    all_images.extend(images)
random.shuffle(all_images)
query_images = all_images[:query_size] 
print('query_images:\n', query_images)

for query in query_images:
    sub_qry = re.match(r'^[A-Za-z_]+(?=\d)', query).group()
    sub_query_dir.append(sub_qry)
print('sub_query_dir:\n', sub_query_dir)


query_processed = []
print("Query Images", query_images)
for img in query_images:
    query_image_path = os.path.join(main_folder, re.match(r'^[A-Za-z_]+(?=\d)', img).group(), img)
    query_image = tf.io.read_file(query_image_path)
    query_image = preprocess_image(query_image)
    query_processed.append(query_image)

print('lenght of query_images', len(query_images))
print('lenght of query_processed', len(query_processed))


query_features = feat_ext(query_processed)
print(len(query_features))

cosine_list = []
rank_list = []

for el in range(len(query_features)):
    similarities = cosine_similarity(query_features[el], np.array(gallery_features).squeeze(axis=1))
  
    cosine_list.append(similarities)
   
    
    rank_list.append(np.argsort(cosine_list[el].squeeze())[-top:])
    print('most_similar_indices:\n', np.argsort(cosine_list[el].squeeze())[-top:])


print()
print('rank_list : ', rank_list)

fig, axs = plt.subplots(query_size, query_size + 1, figsize=(12, 3 * query_size))

for j in range(query_size):
    axs[j, 0].set_title(f'Query: {sub_query_dir[j]}')
    axs[j, 0].imshow(query_processed[j])
    for i in range(len(rank_list)):
        axs[j, i + 1].set_title(f'{[sub_gallery_dir[x] for x in rank_list[j]][i]}')
        axs[j, i + 1].imshow([gallery[x] for x in rank_list[j]][i])
plt.tight_layout()
plt.show()

