!pip install pyarrow==14.0.1

!pip install --upgrade cudf-cu12 ibis-framework

!pip install transformers datasets faiss-cpu -q

# Load model for computing embeddings of the candidate images

from transformers import AutoFeatureExtractor, AutoModel


model_ckpt = "nateraw/vit-base-beans"

extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

hidden_dim = model.config.hidden_size


# Load candidate subset

from datasets import load_dataset


seed = 42
num_samples = 100
dataset = load_dataset("beans", split="train")
candidate_dataset = dataset.shuffle(seed=seed).select(range(num_samples))


# Extract embeddings

def extract_embeddings(image):
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()

dataset_with_embeddings = candidate_dataset.map(lambda example: {'embeddings': extract_embeddings(example["image"])})
dataset_with_embeddings.add_faiss_index(column='embeddings')

# Load test set for querying

test_ds = load_dataset("beans", split="test")


# Select a random sample and run the query
import numpy as np


random_index = np.random.choice(len(test_ds))
query_image = test_ds[random_index]["image"]
query_image

def get_neighbors(query_image, top_k=10):
    qi_embedding = model(**extractor(query_image, return_tensors="pt"))
    qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
    scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', qi_embedding, k=top_k)
    return scores, retrieved_examples


scores, retrieved_examples = get_neighbors(query_image)

from PIL import Image


def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# Images Similar to the Previous Image

images = [query_image]
images.extend(retrieved_examples["image"])


image_grid(images, 1, len(images))

#  Compute Accuracy of the Model

def compute_accuracy(dataset, candidate_dataset, top_k=10):
  correct = 0
  total = 0
  for example in dataset:
    scores, retrieved_examples = get_neighbors(example["image"], top_k)
    retrieved_labels = retrieved_examples['labels']
    if example['labels'] in retrieved_labels:
      correct += 1
    total += 1
  accuracy = correct / total
  print("Accuracy:", accuracy)
  return accuracy

_ = compute_accuracy(test_ds, dataset_with_embeddings)
