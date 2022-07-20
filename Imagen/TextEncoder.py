import torch
from transformers import T5Tokenizer, T5EncoderModel, T5Config

MAX_LENGTH = 256

class TextEncoderT5Based():
    
    def __init__(self, name = 'google/t5-v1_1-small', device='cpu'):
        
        self.device    = device
        self.model     = T5EncoderModel.from_pretrained(name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(name)
        self.embed_dim = T5Config.from_pretrained(name).d_model
        
    def textEncoder(self, texts):
        
        text_encoded = self.tokenizer.batch_encode_plus(texts, return_tensors = "pt", padding = 'longest',
                                                        max_length = MAX_LENGTH, truncation = True)
        
        text_ids = text_encoded.input_ids.to(self.device)
        mask     = text_encoded.attention_mask.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad(): encoded_text = self.model(text_ids, mask).last_hidden_state.detach()
                
        return encoded_text, mask.bool()        


if __name__=='__main__':

    T5 = TextEncoderT5Based()
    print('\n\n==============================================================================\n\n')
    print(T5.textEncoder(['I', 'you', 'yes my'])[0].shape)
    print()
    print(T5.textEncoder(['I', 'yes my'])[0].shape)
    print()
    print(T5.textEncoder(['I', 'you', 'yes my'])[0])
    print()
    print(T5.textEncoder(['I', 'you', 'yes my'])[1])
    print('\nDimension:', T5.embed_dim)

    