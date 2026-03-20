"""
Trains a GPT to add n-digit numbers.
"""

import os
import sys
import json

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from datetime import datetime

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    C.system.work_dir = f'./out/multiplier/{timestamp}/'

    # data
    C.data = MultiplicationDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------


class MultiplicationDataset(Dataset):
    """
    Creates n-digit multiplication problems. For example, if n=2, then an example
    multiplication problem would be to multiply 85 * 50 = 4250. This problem would be
    represented as the following string for the GPT:

    "85500524"

    This is because:
    - we are discarding the * and =, which are not necessary. We just encode the digits
        of the input numbers concatenated together.
    - the result 4250 is encoded backwards (as 0524) to make the multiplication easier to learn for the
        GPT model, because of how the multiplication algorithm works.

    As one more example, the problem 6 * 39 = 234 would be encoded as:

    "063900432"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (2n) (since the product of two n-digit numbers can be up to 2n digits).
    The product is always reversed in the encoding (e.g., 234 becomes 4320 for n=2).
    At test time, we will feed in a multiplication problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (2n) digits (reversed product) correctly.
    """


    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # train/test

        # split up all multiplication problems into either training data or test data
        ndigit = self.config.ndigit
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible multiplication problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]


    def get_vocab_size(self):
        return 10 # digits 0..9


    def get_block_size(self):
        # a, b, a*b,
        # input: n + n digits, output: up to 2n digits (product)
        # so total length: n + n + 2n = 4n
        # but we want to predict the next token, so input is 4n-1
        return 4*self.config.ndigit - 1


    def __len__(self):
        return self.ixes.nelement()


    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # given a problem index idx, first recover the associated a * b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the multiplication problem a * b
        c = a * b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        # product can be up to 2n digits, pad accordingly
        cstr = (f'%0{2*ndigit}d' % c)[::-1] # reverse c to make multiplication easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)


    # construct train and test datasets
    train_dataset = MultiplicationDataset(config.data, split='train')
    test_dataset  = MultiplicationDataset(config.data, split='test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    config.trainer.max_iters = 9001 if config.data.ndigit <=2 else 6001

    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        out_len = ndigit * 2
        mistakes_printed_already = 0
        factors_out = torch.tensor([[10**i for i in range(out_len)][::-1]]).to(trainer.device)
        facotrs_in = torch.tensor([[10**i for i in range(ndigit)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # isolate the first two digits of the input sequence alone
            d1d2 = x[:, :ndigit*2]
            # let the model sample the rest of the sequence
            d1d2d3 = model.generate(d1d2, out_len, do_sample=False) # using greedy argmax, not sampling
            # isolate the last digit of the sampled sequence
            d3 = d1d2d3[:, -out_len:]
            d3 = d3.flip(1) # reverse the digits to their "normal" order
            # decode the integers from individual digits
            d1i = (d1d2[:,:ndigit] * facotrs_in).sum(1)
            d2i = (d1d2[:,ndigit:ndigit*2] * facotrs_in).sum(1)
            d3i_pred = (d3 * factors_out).sum(1)
            d3i_gt = d1i * d2i # manually calculate the ground truth
            # evaluate the correctness of the results in this batch
            correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that %d * %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum(),len(results), 100*rt.mean()

    # iteration callback
    top_score = 0
    train_scores = []
    test_scores = []
    epochs = []
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit] # if ndigit=2 we can afford the whole train set, ow no
            model.eval()
            with torch.no_grad():
                train_score, train_len, train_acc = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score, test_len, test_acc  = eval_split(trainer, 'test',  max_batches=None)
                epochs.append(trainer.iter_num)
            train_scores.append(train_acc.item())
            test_scores.append(test_acc.item())
            scores_df = pd.DataFrame({
                'train_score': train_scores,
                'test_score': test_scores,
                'epoch': epochs
            })
            scores_csv_path = os.path.join(config.system.work_dir, "train_test_scores.csv")
            scores_df.to_csv(scores_csv_path, index=False)

            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
    
    # at the end of training, save train/test scores to a csv file
    scores_df = pd.DataFrame({
        'train_score': train_scores,
        'test_score': test_scores,
        'epoch': epochs
    })
    scores_csv_path = os.path.join(config.system.work_dir, "train_test_scores.csv")
    scores_df.to_csv(scores_csv_path, index=False)
    print(f"Saved train/test scores to {scores_csv_path}")