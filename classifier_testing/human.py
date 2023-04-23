import shutil
import os
import time
import wget
from tqdm import tqdm
from detector import Detector
from utils import write_txt, load_json_file, get_results, load_caption_file, load_standard_json

EXPERIMENT_NAME = "CheckHuman"
CHECK_HUMAN = True
CHECK_BY_IMAGE = False
DATA_FILE = './data/Test_HumanFullSet.json' #Sticking to domain of image captions.
DETECTOR_FILE = "./COCOModel.pt"
IMAGES_TO_RUN = 100

def clean_folder(folder_path):
	if os.path.exists(folder_path):
		shutil.rmtree(folder_path)
	os.makedirs(folder_path)

def run_experiment(detector, experiment_name):
	start_time = time.time()
	out_path = f"./experimental_results/{experiment_name}/"
	adv_text_path = out_path + 'adv_texts/'
	numerical_results_path = out_path + 'results.txt'
	num_changes_path = out_path + 'num_changes.txt'
	clean_folder(out_path)
	clean_folder(adv_text_path)

	print(f"Running Experiment: {experiment_name} ...")
	text_list = load_json_file(DATA_FILE) if 'xl-1542M-k40' in DATA_FILE else \
		load_standard_json(DATA_FILE, CHECK_BY_IMAGE) if ".json" in DATA_FILE else \
			load_caption_file(DATA_FILE, CHECK_BY_IMAGE)
	num_texts = IMAGES_TO_RUN if CHECK_BY_IMAGE else IMAGES_TO_RUN * 5
	text_list = text_list[:num_texts]
	_range = tqdm(range(len(text_list)))

	for i in _range:
		text_to_use = detector.tokenizer.decode(
			detector.tokenizer.encode(text_list[i], max_length=detector.tokenizer.max_len))[3:-4]

		adv_text, num_changes = (text_to_use, 0)

		write_txt(f"{adv_text_path}{i}.txt", adv_text)
		probs = detector.predict(adv_text)
		human_prob = probs[1]
		_range.set_description(f"{i} | {human_prob:.2f}")

		with open(numerical_results_path, 'a') as f:
			f.write(f"{human_prob:.2f} ")
		with open(num_changes_path, 'a') as f:
			f.write(f"{num_changes} ")
	end_time = time.time()

	print('Time to complete experiment (minutes):', (end_time - start_time) / 60.)

if __name__ == '__main__':
	if not os.path.exists(DETECTOR_FILE):
		print("Downloading detector...")
		wget.download("https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt")
	detector = Detector(DETECTOR_FILE)
	run_experiment(detector, EXPERIMENT_NAME)
	get_results(EXPERIMENT_NAME, CHECK_HUMAN)
