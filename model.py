import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
# import matplotlib.pyplot as plt
# from torchvision.transforms.functional import to_pil_image

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)      # self.linear = Linear(in_features=2048, out_features=256, bias=True)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)          # features.shape = ([128, 2048, 1, 1])
        features = features.reshape(features.size(0), -1)   # features.shape = ([128, 2048])
        features = self.bn(self.linear(features))           # features.shape = ([128,256])
        return features

    def layered_photo(self, image):
        # because the image is transformed in the dataloader, we will transform the image here aswell 
        transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
        
        # we will save the conv layer weights in this list
        model_weights =[]
        #we will save the 151 conv layers in this list
        conv_layers = []
        # get all the model children as list
        model_children = list(self.resnet)
        print(model_children)
        #counter to keep count of the conv layers
        counter = 0
        #append all the conv layers and their respective wights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter+=1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter+=1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolution layers: {counter}")
        # for conv in conv_layers:
        #     print(conv)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = self.resnet.to(device)

        plt.imshow(image)
        plt.show()        # image before transformation
        image = transform(image)
        # to_pil_image(image).show()        # image after transformation
        print(f"Image shape before: {image.shape}")
        image = image.unsqueeze(0)
        print(f"Image shape after: {image.shape}")
        image = image.to(device)

        outputs = []
        names = []
        for layer in conv_layers[0:]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))
        print(len(outputs))
        # print feature_maps
        for feature_map in outputs:
            print(feature_map.shape)      # print feature shape (batch_size, )
        

        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        # for fm in processed:
        #     print(fm.shape)       # print image shape (height, width)

        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(13, 12, i+1)        # len(processed) is 151 therefore it's needed an amount of at least 151 subplots
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(names[i].split('(')[0], fontsize=30)
        plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
        

        



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)           # self.embed = Embedding(9948, 256) 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)      # self.lstm = LSTM(256, 512, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)            # self.linear = Linear(in_features=512, out_features=9948, bias=True)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)       # embeddings = [batch_size, max_seq_length, embed_size]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)      # embeddings = [batch_size, max_seq_length + 1, embed_size]
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])       # outputs.shape = ([])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids