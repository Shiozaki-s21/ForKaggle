import pandas as pd


class CheckAccuracy:
    def __init__(self):
       self.ans_csv = pd.read_csv('../Titanic/titanic_data/gender_submission.csv')
       self.ans_survived = self.ans_csv.Survived

       self.result = pd.read_csv('result.csv')

    def check(self):
        count = 0

        for i in range(self.ans_csv.Survived.shape[0]):
            if self.ans_csv.Survived[i] == self.result.Survived[i]:
                count = count + 1

        print('correct number: ' + str(count))
        print('correct rate' + str(count / self.ans_csv.shape[0]))


checker = CheckAccuracy()
checker.check()
