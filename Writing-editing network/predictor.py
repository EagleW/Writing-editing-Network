import torch
import numpy as np


class Predictor(object):
    def __init__(self, model, vectorizer):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.vectorizer = vectorizer

    def predict(self, src_seq, num_exams):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        torch.set_grad_enabled(False)
        text = []
        for tok in src_seq:
            if tok in self.vectorizer.word2idx:
                text.append(self.vectorizer.word2idx[tok])
            else:
                text.append(3)

        input_variable = torch.LongTensor(text).view(1, -1)
        if torch.cuda.is_available():
            input_variable = input_variable.cuda()

        input_lengths = [len(src_seq)]

        prev_generated_seq = None
        outputs = []
        for i in range(num_exams):
            _, _, other = \
                self.model(input_variable, prev_generated_seq, input_lengths)
            length = other['length'][0]

            tgt_id_seq = [other['sequence'][di][0].item() for di in range(length)]
            tgt_seq = [self.vectorizer.idx2word[tok] for tok in tgt_id_seq]
            output = ' '.join([i for i in tgt_seq if i != '<PAD>' and i != '<EOS>' and i != '<SOS>'])
            outputs.append(output)
            prev_generated_seq = torch.LongTensor(tgt_id_seq).view(1, -1)
            if torch.cuda.is_available():
                prev_generated_seq = prev_generated_seq.cuda()
        return outputs

    def predict_batch(self, source, input_lengths, num_exams):
        torch.set_grad_enabled(False)
        output_seq = []
        input_variables = source
        for i in range(source.size(0)):
            title_id_seq = [input_variables[i][di].item() for di in range(input_lengths[i])]
            title_seq = [self.vectorizer.idx2word[tok] for tok in title_id_seq]
            title = ' '.join([k for k in title_seq if k != '<PAD>' and k != '<EOS>' and k != '<SOS>'])
            output_seq.append([title])
        prev_generated_seq = None
        for k in range(num_exams):
            _, _, other = \
                self.model(input_variables, prev_generated_seq, input_lengths)
            length = other['length']
            sequence = torch.stack(other['sequence'], 1).squeeze(2)
            prev_generated_seq = self._mask(sequence)
            for i in range(len(length)):
                opt_id_seq = [other['sequence'][di][i].item() for di in range(length[i])]
                opt_seq = [self.vectorizer.idx2word[tok] for tok in opt_id_seq]
                output = ' '.join([k for k in opt_seq if k != '<PAD>' and k != '<EOS>' and k != '<SOS>'])
                output_seq[i].append(output)
        return output_seq

    # Mask variable
    def _mask(self, prev_generated_seq):
        prev_mask = torch.eq(prev_generated_seq, 1).cpu().data.numpy()
        lengths = np.argmax(prev_mask,axis=1)
        max_len = prev_generated_seq.size(1)
        mask = []
        for i in range(prev_generated_seq.size(0)):
            if lengths[i] == 0:
                mask_line = [0] * max_len
            else:
                mask_line = [0] * lengths[i]
                mask_line.extend([1] * (max_len - lengths[i]))
            mask.append(mask_line)
        mask = torch.ByteTensor(mask)
        if torch.cuda.is_available():
            mask = mask.cuda()
        return prev_generated_seq.data.masked_fill_(mask, 0)

    def preeval_batch(self, test_loader, abs_len, num_exams):
        torch.set_grad_enabled(False)
        refs = {}
        cands = []
        tmp = []
        for i in range(num_exams):
            cands.append({})
            tmp.append({})
        i = 0
        for batch_idx, (source, target, input_lengths) in enumerate(test_loader):
            input_variables = source
            input_lengths = input_lengths.tolist()
            prev_generated_seq = None
            for k in range(num_exams):
                _, _, other = \
                    self.model(input_variables, prev_generated_seq, input_lengths)
                length = other['length']
                sequence = torch.stack(other['sequence'], 1).squeeze(2)
                prev_generated_seq = self._mask(sequence)
                for j in range(len(length)):
                    out_seq = [other['sequence'][di][j] for di in range(length[j])]
                    out = self.prepare_for_bleu(out_seq)
                    tmp[k][j] = out
            for j in range(source.size(0)):
                i += 1
                ref = self.prepare_for_bleu(target[j])
                refs[i] = [ref]
                for k in range(num_exams):
                    cands[k][i] = tmp[k][j]
            if i % 100 == 0:
                print("Percentages:  %.4f" % (i/float(abs_len)))
        return cands, refs


    def prepare_for_bleu(self, sentence):
        sent=[x.item() for x in sentence if x.item() != 0 and x.item() != 1 and x.item() != 2]
        sent = ' '.join([str(x) for x in sent])
        return sent

    def predict_seq_title(self, title, sec_seq, num_exams):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        torch.set_grad_enabled(False)
        text = []
        for tok in title:
            if tok in self.vectorizer.word2idx:
                text.append(self.vectorizer.word2idx[tok])
            else:
                text.append(3)

        input_variable = torch.LongTensor(text).view(1, -1)
        if torch.cuda.is_available():
            input_variable = input_variable.cuda()

        input_lengths = [len(title)]

        text = []
        for tok in sec_seq:
            if tok in self.vectorizer.word2idx:
                text.append(self.vectorizer.word2idx[tok])
            else:
                text.append(3)

        prev_generated_seq = torch.LongTensor(text).view(1, -1)
        if torch.cuda.is_available():
            prev_generated_seq = prev_generated_seq.cuda()

        outputs = []
        for i in range(num_exams):
            _, _, other = \
                self.model(input_variable, prev_generated_seq, input_lengths)
            length = other['length'][0]

            tgt_id_seq = [other['sequence'][di][0].item() for di in range(length)]
            tgt_seq = [self.vectorizer.idx2word[tok] for tok in tgt_id_seq]
            output = ' '.join([i for i in tgt_seq if i != '<PAD>' and i != '<EOS>' and i != '<SOS>'])
            outputs.append(output)
            prev_generated_seq = torch.LongTensor(tgt_id_seq).view(1, -1)
            if torch.cuda.is_available():
                prev_generated_seq = prev_generated_seq.cuda()
        return outputs
