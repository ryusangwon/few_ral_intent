# Copyright 2020, Salesforce.com, Inc.

class IntentPredictor:
    def __init__(self,
                 tasks = None):

        self.tasks = tasks

    def predict_intent(self,
                       input: str):
        raise NotImplementedError

class DnncIntentPredictor(IntentPredictor):
    def __init__(self,
                 model,
                 tasks=None):

        super().__init__(tasks)

        self.model = model

    def predict_intent(self,
                       input: str):

        nli_input = []
        for t in self.tasks:
            for e in t['examples']:
                nli_input.append((input, e))

        assert len(nli_input) > 0

        results = self.model.predict(nli_input)
        maxScore, maxIndex = results[1][:, 0].max(dim=0)

        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        index = -1
        for t in self.tasks:
            for e in t['examples']:
                index += 1

                if index == maxIndex:
                    intent = t['task']
                    matched_example = e

        return intent, maxScore, matched_example