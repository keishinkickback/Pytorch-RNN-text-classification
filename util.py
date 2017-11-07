import csv

# import nltk

LABEL_TO_INDEX = {
    'business':                  0,
    'computers':                 1,
    'culture-arts-entertainment':2,
    'education-science':         3,
    'engineering':               4,
    'health':                    5,
    'politics-society':          6,
    'sports':                    7
}

def create_tsv_file(path_in, path_out):

    with open(path_in,'r') as f, open(path_out,'w') as fw:
        writer = csv.writer(fw, delimiter='\t')
        writer.writerow(['label','body'])
        for line in f:
            tokens = [x.lower() for x in line.split()]
            label = LABEL_TO_INDEX[tokens[-1]]
            body = ' '.join(tokens[:-1])
            writer.writerow([label, body])


def _tokenize(text):
    # return [x.lower() for x in nltk.word_tokenize(text)]
    return [ x.lower() for x in text.split() ]


''' from https://github.com/pytorch/examples/blob/master/imagenet/main.py'''
class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 8 epochs"""
    lr = lr * (0.1 ** (epoch // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

