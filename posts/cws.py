from torch import nn
import torch
# from fastNLP.modules.encoder.transformer import TransformerEncoder
from fastNLP.models import StarTransEnc
from fastNLP.modules import ConditionalRandomField, 
from fastNLP import seq_len_mask
# from fastNLP.modules.decoder.crf import allowed_transitions

class TransformerCWS(nn.Module):
    def __init__(self, vocab_num, max_len, embed_dim=100, bigram_vocab_num=None, 
                 bigram_embed_dim=100, num_bigram_per_char=None,
                 hidden_size=200, embed_drop_p=0.3, num_layers=2, num_heads=6, tag_size=4):
        super().__init__()

        input_size = embed_dim
        if bigram_vocab_num:
            self.bigram_embedding = nn.Embedding(bigram_vocab_num, bigram_embed_dim)
            input_size += num_bigram_per_char*bigram_embed_dim

        self.drop = nn.Dropout(embed_drop_p, inplace=True)

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.transformer = StarTransEnc(nn.Embedding(vocab_num, embed_dim), num_layers=num_layers, 
                                        hidden_size=hidden_size, num_head=num_heads, head_dim=32,
                                        emb_dropout=0.3, dropout=0.1, max_len=max_len)
        self.fc2 = nn.Linear(hidden_size, tag_size)

        # allowed_trans = allowed_transitions({0:'b', 1:'m', 2:'e', 3:'s'}, encoding_type='bmes')
        allowed_trans = None
        self.crf = ConditionalRandomField(num_tags=tag_size, include_start_end_trans=False,
                                          allowed_transitions=allowed_trans)

    def forward(self, chars, target, seq_lens, bigrams=None):
        masks = seq_len_to_mask(seq_lens)
        batch_size = x.size(0)
        length = x.size(1)
        if hasattr(self, 'bigram_embedding'):
            bigrams = self.bigram_embedding(bigrams) # batch_size x seq_lens x per_char x embed_size
            x = torch.cat([x, bigrams.view(batch_size, length, -1)], dim=-1)
        feats, _ = self.transformer(chars, masks)
        feats = self.fc2(feats)
        losses = self.crf(feats, target, masks.float())

        pred_dict = {}
        pred_dict['seq_lens'] = seq_lens
        pred_dict['loss'] = torch.mean(losses)

        return pred_dict

    def predict(self, chars, seq_lens, bigrams=None):
        masks = seq_len_to_mask(seq_lens)

        x = self.embedding(chars)
        batch_size = x.size(0)
        length = x.size(1)
        if hasattr(self, 'bigram_embedding'):
            bigrams = self.bigram_embedding(bigrams) # batch_size x seq_lens x per_char x embed_size
            x = torch.cat([x, bigrams.view(batch_size, length, -1)], dim=-1)
        self.drop(x)
        x = self.fc1(x)
        feats = self.transformer(x, masks)
        feats = self.fc2(feats)

        probs = self.crf.viterbi_decode(feats, masks, get_score=False)

        return {'pred': probs, 'seq_lens':seq_lens}


if __name__ == "__main__":
    from fastNLP import Batch 
    from fastNLP import RandomSampler
    from train_star import load_ppl2014_fastway

    ds, word_v, tag_v = load_ppl2014_fastway('/home/darktower/nlp_exp/data/ppl2014')
    train, test = ds.split(0.8)
    del ds, test

    data_iterator = Batch(train, sampler=RandomSampler(), batch_size =64, as_numpy=False)
    for batch_x, batch_y in data_iterator:
        break

    cws = TransformerCWS(len(word_vocab), 350, tag_size=len(tag_v))
    print('Loss is: ', cws(batch_x['words'], batch_y['target'], batch_x['seq_len']))