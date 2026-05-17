import pycrfsuite


def word_features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    wl = word.lower()

    features = {
        "bias": 1.0,
        "word.lower": wl,
        "word[-3:]": wl[-3:],
        "word[-2:]": wl[-2:],
        "word[:3]": wl[:3],
        "word[:2]": wl[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.isalpha": word.isalpha(),
        "word.length": len(word),
        "word.has_hyphen": "-" in word,
        "word.has_period": "." in word,
        "word.has_digit": any(c.isdigit() for c in word),
        "word.has_upper": any(c.isupper() for c in word),
        "word.all_lower": word.islower(),
        "pos": pos,
        "pos[:2]": pos[:2],
    }

    if i > 0:
        prev = sent[i - 1][0]
        prev_pos = sent[i - 1][1]
        features.update({
            "-1:word.lower": prev.lower(),
            "-1:word.istitle": prev.istitle(),
            "-1:word.isupper": prev.isupper(),
            "-1:word[-3:]": prev.lower()[-3:],
            "-1:pos": prev_pos,
            "-1:pos[:2]": prev_pos[:2],
        })
    else:
        features["BOS"] = True

    if i > 1:
        prev2 = sent[i - 2][0]
        features.update({
            "-2:word.lower": prev2.lower(),
            "-2:word.istitle": prev2.istitle(),
        })

    if i < len(sent) - 1:
        nxt = sent[i + 1][0]
        nxt_pos = sent[i + 1][1]
        features.update({
            "+1:word.lower": nxt.lower(),
            "+1:word.istitle": nxt.istitle(),
            "+1:word.isupper": nxt.isupper(),
            "+1:word[-3:]": nxt.lower()[-3:],
            "+1:pos": nxt_pos,
            "+1:pos[:2]": nxt_pos[:2],
        })
    else:
        features["EOS"] = True

    if i < len(sent) - 2:
        nxt2 = sent[i + 2][0]
        features.update({
            "+2:word.lower": nxt2.lower(),
            "+2:word.istitle": nxt2.istitle(),
        })

    return features


def sent_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]


def sent_labels(sent):
    return [tag for _, _, tag in sent]


class CRFModel:
    def __init__(self, algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100):
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.tagger = None

    def train(self, train_sents, model_path):
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params({
            "c1": self.c1,
            "c2": self.c2,
            "max_iterations": self.max_iterations,
            "feature.possible_transitions": True,
        })

        for sent in train_sents:
            features = sent_features(sent)
            labels = sent_labels(sent)
            trainer.append(features, labels)

        trainer.train(str(model_path))

    def load(self, model_path):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(model_path))

    def predict(self, sentences):
        results = []
        for sent in sentences:
            features = sent_features(sent)
            tags = self.tagger.tag(features)
            results.append(tags)
        return results

    def predict_tokens(self, tokens):
        dummy = [(t, _guess_pos(t), "O") for t in tokens]
        features = sent_features(dummy)
        return self.tagger.tag(features)

    def predict_tokens_with_confidence(self, tokens):
        dummy = [(t, _guess_pos(t), "O") for t in tokens]
        features = sent_features(dummy)
        self.tagger.set(features)
        tags = self.tagger.tag()
        marginals = [self.tagger.marginal(tag, i) for i, tag in enumerate(tags)]
        return tags, marginals


def _guess_pos(token):
    if token[0].isupper() and len(token) > 1:
        return "NNP"
    if token.isdigit():
        return "CD"
    if token in {"the", "a", "an"}:
        return "DT"
    if token in {"is", "was", "are", "were", "am", "be"}:
        return "VBZ"
    if token in {"in", "on", "at", "to", "for", "of", "with", "from"}:
        return "IN"
    if token in {"my", "your", "his", "her", "its", "our", "their"}:
        return "PRP$"
    if token in {"i", "he", "she", "it", "we", "they", "you"}:
        return "PRP"
    return "NN"
