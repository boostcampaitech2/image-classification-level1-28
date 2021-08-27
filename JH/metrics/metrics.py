from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class Validate():
    def __init__(self, loader, length, model, config, criterions):
        self.model = model
        self.loader = loader
        self.config = config
        self.length = length
        self.mask_criterion, self.gender_criterion, self.age_criterion = criterions
        self.labels_to_class = {}
        it = [(m, g, a) for m in [0,1,2] for g in [0, 1] for a in [0, 1, 2]]
        for i, (m, g, a) in enumerate(it):
            self.labels_to_class[(m, g, a)] = i
    def check():
        device = torch.device("cuda:0")
        model.eval()
        val_loss = 0.0
        counter = 0
        y_true = []
        y_predicted = []

        m_acc = []
        g_acc = []
        a_acc = []
        with torch.no_grad():
            for (inputs, (m, g, a)) in self.loader:
                counter += 1

                for mask, gender, age in zip(m, g, a):
                    answer = self.labels_to_class[(mask.item(), gender.item(), age.item())]
                    y_true.append(answer)

                inputs = inputs.to(device=device)
                m = m.to(device)
                g = g.to(device)
                a = a.to(device)

                m_pred, g_pred, a_pred = model(inputs)

                m_loss = self.mask_criterion(m_pred, m)
                g_loss = self.gender_criterion(g_pred, g)
                a_loss = self.age_criterion(a_pred, a) # data imbalance

                loss = (g_loss+a_loss+m_loss)

                val_loss += loss.item()

                m_argmax = m_pred.detach().cpu().numpy().argmax(1)
                g_argmax = g_pred.detach().cpu().numpy().argmax(1)
                a_argmax = a_pred.detach().cpu().numpy().argmax(1)

                m_acc.append(accuracy_score(m_argmax, m.detach().cpu().numpy()))
                g_acc.append(accuracy_score(g_argmax, g.detach().cpu().numpy()))
                a_acc.append(accuracy_score(a_argmax, a.detach().cpu().numpy()))

                for mask, gender, age in zip(m_argmax, g_argmax, a_argmax):
                    predicted = self.labels_to_class[(mask.item(), gender.item(), age.item())]
                    y_predicted.append(predicted)


        cm = confusion_matrix(y_true, y_predicted)
        F1 = []
        for c in range(18):
            precision = cm[c][c] / np.sum(cm, axis=0)[c]
            recall = cm[c][c] / np.sum(cm, axis=1)[c]
            F1.append(2 * precision * recall / (precision + recall))
        macro_F1 = np.mean(F1)

        s = 0
        for c in range(18):
            s += cm[c][c]

        print("< VALIDATION >")
        print("*"*73)
        print("Validation Loss :", val_loss/counter)
        print("-"*73)
        print("Total Accuracy")
        print(accuracy_score(y_true, y_predicted) * 100, "%")
        print("-"*73)
        print("Class Accuracy")
        print("Mask   :", np.mean(m_acc)*100, "%")
        print("Gender :", np.mean(g_acc)*100, "%")
        print("Age    :", np.mean(a_acc)*100, "%")
        print("-"*73)
        print("Confusion Matrix")
        for row in cm:
            for c in row:
                print(str(c).ljust(4), end='')
            print()
        print("-"*73)
        print("Validation F1 score :" , macro_F1)
        for c, f in enumerate(F1):
            print("Class", c, ":", f)

        if model.best_f1 < macro_F1:
            model.best_f1 = macro_F1
            torch.save(model.state_dict(), '/opt/ml/weights/{}/{:.4f}.pt'.format(config['model'], model.best_f1))
            print("model saved!")
        print("*"*73)
        print()
        wandb.log({
            "Validation Loss" : val_loss/counter, 
            "Validation Total Accuracy" :accuracy_score(y_true, y_predicted) * 100, 
            "Validation F1" : macro_F1,
            "Mask Accuracy" : np.mean(m_acc)*100,
            "Gender Accuracy" : np.mean(g_acc)*100,
            "Age Accuracy" : np.mean(a_acc)*100,
            "Mask Loss" : m_loss.item(),
            "Gender Loss" : g_loss.item(),
            "Age Loss" : a_loss.item(),
        })

        model.train()