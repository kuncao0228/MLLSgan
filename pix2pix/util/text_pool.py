import random
import torch


class TextPool():
    """This class implements an image buffer that stores previously generated texts.

    This buffer enables us to update discriminators using a history of generated texts
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the TextPool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.text_embeddings = []

    def query(self, embeddings):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return embeddings
        return_embeddings = []
        for embed in embeddings:
            #image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.text_embeddings.append(embed)
                return_embeddings.append(embed)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.text_embeddings[random_id].clone()
                    self.text_embeddings[random_id] = embed
                    return_embeddings.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_embeddings.append(embed)
        return_embeddings = torch.cat(return_embeddings, 0)   # collect all the images and return
        return return_embeddings
