import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class Detector(object):

	def __init__(self, detector_file_name):

		print('Initializing Detector...')

		data = torch.load(detector_file_name)
		self.model = RobertaForSequenceClassification.from_pretrained('roberta-large')
		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
		if detector_file_name == "./detector-large.pt":
			self.model.load_state_dict(data['model_state_dict'])
		else:
			self.model.load_state_dict(data)
		self.model.eval()


	def predict(self, txt):

		tokens = self.tokenizer.encode(txt, max_length=self.tokenizer.max_len)
		#tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
		tokens = torch.Tensor(tokens)

		tokens = tokens.unsqueeze(0).long()

		mask = torch.ones_like(tokens).long()

		logits = self.model(tokens, attention_mask=mask)

		probs = logits[0].softmax(dim=-1)

		probs = probs.detach().cpu().flatten().numpy()

		return probs

	def get_result(self, txt):
		p = self.predict(txt)
		prob = np.max(p)
		result = np.argmax(p)
		if result == 1:
			result = 'Human'
		else:
			result = 'Machine'
		print(result, ' | ', prob)


if __name__ == '__main__':

	text = 'adversarial machine learning'

	det = Detector()
	p = det.predict(text)
	prob = np.max(p)
	result = np.argmax(p)
	if result == 1:
		result = 'Human'
	else:
		result = 'Machine'
	print(result, ' | ', prob)



