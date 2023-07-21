import numpy as np
import torch
import torch.nn as nn
from torch import hub

import vggish_input, vggish_params


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty((vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),dtype=torch.float,)
        self.pca_means = torch.empty((vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float)

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)
        
        state_dict = torch.load('weights/vggish_pca_params-970ea276.pth')
        state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float)
        state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float)

        self.load_state_dict(state_dict)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (embeddings_batch.shape,)
        assert (embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()
        
        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL)
        
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round((clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)* (255.0/ (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)))


        return quantized_embeddings

    def forward(self, x):
        x = self.postprocess(x)
        return x
    



class VGGish(nn.Module):
    def __init__(self, selected_exits, preprocess=False, postprocess=True):
        super().__init__()

        self.selected_exits = selected_exits
        self.preprocess = preprocess
        self.postprocess = postprocess

        self.relu = nn.ReLU(inplace=True)

        if self.selected_exits[-1] >=1:
            self.conv1_block_br1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.pool1_block_br1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.selected_exits[-1] >=2:
            self.conv2_block_br2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool2_block_br2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.selected_exits[-1] >=3:
            self.conv3_block_br3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv4_block_br3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool4_block_br3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.selected_exits[-1] >=4:
            self.conv5_block_br4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv6_block_br4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.pool6_block_br4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        if 1 in self.selected_exits:
            self.embeddings_exit_br1 = nn.Sequential(
                nn.Linear(64*48*32, 128),
                nn.ReLU(True),
                # nn.Linear(4096, 4096),
                # nn.ReLU(True),
                # nn.Linear(4096, 128),
                # nn.ReLU(True), 
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 2 in self.selected_exits:
            self.embeddings_exit_br2 = nn.Sequential(
                nn.Linear(128*24*16, 128),
                nn.ReLU(True),
                # nn.Linear(4096, 4096),
                # nn.ReLU(True),
                # nn.Linear(4096, 128),
                # nn.ReLU(True), 
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 3 in self.selected_exits:
            self.embeddings_exit_br3 = nn.Sequential(
                nn.Linear(256*12*8, 128),
                nn.ReLU(True),
                # nn.Linear(4096, 4096),
                # nn.ReLU(True),
                # nn.Linear(4096, 128),
                # nn.ReLU(True), 
                Postprocessor(),
                torch.nn.Linear(128, 527))
        
        if 4 in self.selected_exits:
            self.embeddings_exit_br4 = nn.Sequential(
                nn.Linear(512*4*6, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 128),
                nn.ReLU(True), 
                Postprocessor(),
                torch.nn.Linear(128, 527))
            


    def forward(self, x, fs=None):
        out_vector_list = []
        
        ### block1
        if self.selected_exits[-1] >=1:
            x = self.conv1_block_br1(x)
            x = self.relu(x)
            x = self.pool1_block_br1(x)
            
        # exit1
        if 1 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br1(temp))
        
        
        ### block2
        if self.selected_exits[-1] >=2:
            x = self.conv2_block_br2(x) 
            x = self.relu(x)
            x = self.pool2_block_br2(x) 
            
        # exit2
        if 2 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br2(temp))
        
        ### block3
        if self.selected_exits[-1] >=3:
            x = self.conv3_block_br3(x)
            x = self.relu(x)
            x = self.conv4_block_br3(x)
            x = self.relu(x)
            x = self.pool4_block_br3(x)
            
        # exit3
        if 3 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br3(temp))
        
        ### block4
        if self.selected_exits[-1] >=4:
            x = self.conv5_block_br4(x)
            x = self.relu(x)
            x = self.conv6_block_br4(x)
            x = self.relu(x)
            x = self.pool6_block_br4(x)
            
            
        # exit4
        if 4 in self.selected_exits:
            temp = torch.transpose(x, 1, 3)
            temp = torch.transpose(temp, 1, 2)
            temp = temp.contiguous().view(temp.size(0), -1)
            out_vector_list.append(self.embeddings_exit_br4(temp))
        
        return out_vector_list

