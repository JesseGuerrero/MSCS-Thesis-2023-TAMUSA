import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import subprocess

def load_txt(filename):
	lines = ''
	with open(filename, encoding="utf-8") as fp:
		line = fp.readline()
		while line:
			l = line.strip()
			lines += l
			lines += '\n'
			line = fp.readline()
	fp.close()
	return lines

def write_txt(filename, string):
	with open(filename, 'w', encoding="utf-8") as fp:
		fp.write(string)
	fp.close()


def load_misspelling_dict(fname):
	with open(fname, 'r') as fp:
		d = json.load(fp)
	fp.close()
	return d


def load_json_file(filename):

	with open(filename, 'r') as json_file:
		json_list = list(json_file)
	json_file.close()

	texts = [json.loads(x)['text'] for x in json_list]

	return texts

def load_standard_json(filename, check_by_image):
	data = {}
	with open(filename, 'r') as json_file:
		data = dict(json.load(json_file))
	texts= []
	if check_by_image is True:
		dataDict = {}
		for img_name, captions in data.items():
			text = ""
			for caption in captions:
				text = text + caption + " "
			texts.append(text)
	if check_by_image is False:
		for img_name, captions in data.items():
			for caption in captions:
				texts.append(caption)
	return texts

def load_caption_file(filename, check_by_image):
	with open(filename, 'r') as file:
		textLines = file.readlines()
	file.close()
	texts = []
	if check_by_image:
		dictionaryByImage = {}
		for line in textLines:
			if line.split(":")[0] in dictionaryByImage:
				dictionaryByImage[line.split(":")[0]] = dictionaryByImage[line.split(":")[0]] + line.split(":")[1].replace("\n", " ")
			if line.split(":")[0] not in dictionaryByImage:
				dictionaryByImage[line.split(":")[0]] = line.split(":")[1].replace("\n", " ")
		texts = list(dictionaryByImage.values())
		# print(dictionaryByImage)
	if not check_by_image:
		for line in textLines:
			texts.append(line.split(":")[1])
	return texts

def get_results(experiment_name, check_human):
	print('Results for ' + experiment_name)
	results_pth = './experimental_results/'+experiment_name
	asr_pth = results_pth+'/results.txt'
	num_ch_pth = results_pth+'/num_changes.txt'

	probs = load_txt(asr_pth).split(' ')
	num_ch = load_txt(num_ch_pth).split(' ')
	probs = np.float32(np.asarray(probs))

	num_human = 0
	num_changed = 0
	pths = glob.glob('./experimental_results/'+experiment_name+'/adv_texts/*.txt')
	avg = 0
	for pth in pths:
		txt_i = load_txt(pth)
		avg += len(txt_i)
	print(avg / len(pths))


	for i in range(probs.size):
		if np.float32(probs[i]) >= 0.5:
			num_human+=1
		else:
			pass
		num_changed+=np.float32(num_ch[i])#How many characters cut off

	with open(asr_pth, "a") as f:
		f.write('\nAverage confidence:' + str(np.mean(probs)))
		f.write('\nNumber human:' + f'Human: {num_human}, Mutation: {probs.size - num_human}')
		if check_human:
			f.write('\nDetector accuracy:' + str(num_human / probs.size))
		if not check_human:
			f.write('\nDetector accuracy:' + str(1. - (num_human / probs.size)))
		f.write('\nAverage number of changes:' + str(num_changed/len(num_ch)))
		f.write('\nNumber of Attacks Run:' + str(probs.size))
	print('Average confidence:', np.mean(probs))
	print('Number human:', f'Human: {num_human}, Mutation: {probs.size - num_human}')
	if check_human:
		print('Detector accuracy:', (num_human / probs.size))
	if not check_human:
		print('Detector accuracy:', 1. - (num_human / probs.size))
	print('Average number of changes:', num_changed / len(num_ch))
	print('Number of Attacks Run:', probs.size)



def get_graph_data(exp_name):
	exp_name_list = exp_name.split('_')
	pths = glob.glob('./experimental_results/{}'.format(exp_name))
	ext = './xperimental_results'
	pths = [pth for pth in pths if '_1.0_' not in pth]
	x = np.arange(0.00, 0.0525, 0.0025)
	raw = []
	for i in range(x.size):
		probs = np.float32(np.asarray(load_txt(ext+'/{}_{}_{}'.format(exp_name_list[0], str(x[i]), exp_name_list[-1])+'/results.txt').split(' ')))
		raw.append(probs)
	raw = np.asarray(raw)
	success = np.uint8(raw+0.5)
	num_human = np.sum(success, axis=-1)
	asrs = 1. - (num_human / probs.size)
	fig = plt.figure()
	plt.scatter(x, asrs, s=10)
	plt.plot(x, asrs)
	plt.xlabel('Max. Pct. of Text Sample Characters Replaced')
	plt.ylabel('Detector Accuracy')
	plt.title(r'Detector Accuracy vs. Max. Pct. of Text Sample Characters Replaced')
	plt.xlim(-0.002, 0.052)
	plt.savefig('all_graph.png')
	plt.show()
	

	return x, np.asarray(asrs)

def runcmd(cmd, verbose = False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


