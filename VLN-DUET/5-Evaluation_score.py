import jsonlines
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

anno_file = "datasets/R2R/annotations/pretrain/train_add_heading_caption_add_elevation_baichuan_generate_instructions.jsonl"
rouge = Rouge()
spice_obj  = Spice()
cider = Cider()


class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
ref = {}
gt = {}
with jsonlines.open(anno_file, 'r') as f:
    for index, item in enumerate(f):
        instructions = item["instructions"]
        generate_instructions = item["generate_instructions"]
        gt[index]= [''.join(str(i) for i in generate_instructions)]
        ref[index]=[''.join(str(i) for i in instructions)]

scorer = Scorer(ref, gt)
scorer.compute_scores()
print()