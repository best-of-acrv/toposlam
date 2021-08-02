'''
The code is based on the followin paper:
  "Scalable place recognition under appearance change for autonomous driving." Doan, Anh-Dzung, et al. ICCV 2019
'''

from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import sparse


def graph_update(node_num, image_ids, trans_mat, feature, image_id, max_time_step):
    '''
    This function assume that all the added images are consecutive video frames and they are added in a temporal order.

    Input:
           node_num:         the number of previously added image features, this is also the number of the features in
                             the database, we simply get this value from the FAISS index structure using its 'ntotal'
                             attribute

           image_ids:        a list of length N1, correspond to all the ids of the image(features) in the 'database'

           trans_mat:        a 2D matrix of N1 x N1, the transition matrix of the all the features in 'database',
                             defined in Sec. 5.1.  dtype: scipy parse matrix

           feature:          row vector(s) of size n x D, the new deep vlad feature(s) to be added to the 'database',
                             extracted from the image(s) to be added, dtype: numpy array

           image_id:         a list of length n, the corresponding id(s) of the image(s) to be added, dtype: list

           max_time_step:    the maximum step size for affinity matrix computation, defined in Sec. 5.1, dtype: int

    Output:
           image_ids_updated:    the updated image id list of length N2 where N2 = N1 + n
           trans_mat_updated:    the updated transition matrix of size N2 x N2
    '''

    N1 = node_num  # N1 features in the database before update
    n = feature.shape[0]  # n features to be added to the database
    N2 = N1 + n  # there will be N2 features in the database after update

    # update state transition matrix
    start_row = max([0, N1 - max_time_step])  # from which row the node transition probabilities need to be updated

    trans_mat_keep = trans_mat[0: start_row, :]
    zeros_block = sparse.csr_matrix(np.zeros([trans_mat_keep.shape[0], N2 - trans_mat_keep.shape[1]]))
    trans_mat_keep = sparse.hstack([trans_mat_keep, zeros_block])  # extend the column number of the kept block to N2

    trans_mat_new = update_state_transition_matrix(start_row, N2, max_time_step)
    trans_mat_updated = sparse.vstack([trans_mat_keep, trans_mat_new])
    trans_mat_updated = sparse.csr_matrix(trans_mat_updated)

    # update image_ids
    image_ids_updated = image_ids + image_id

    return image_ids_updated, trans_mat_updated


def update_state_transition_matrix(start_row, total_row, max_time_step):
    '''
    This function incrementally updates the state transition matrix with the newly added images(features)
    Input:
          start_row:         the index of the row from which the node transition probabilities that need to be updated
          total_row:         the total number of rows in the transition matrix after update
          max_time_step:     the maximum step size to construct the edges in the graph

    Output:
          trans_mat_new:     a matrix of size (total_row - start_row + 1) x total_row, each row sum to 1
    '''

    gamma = 5
    gamma = gamma**2

    trans_mat_new = np.zeros([total_row - start_row, total_row])
    for row in range(total_row - start_row):
        col_idx = list(range(np.max([0, row + start_row - max_time_step]),
                             np.min([total_row - 1, row + start_row + max_time_step]) + 1))
        col_value = np.array(col_idx)
        trans_mat_new[row, col_idx] = np.exp(- (row - col_value + start_row)**2 / gamma)

    # normalization each row to sum 1
    trans_mat_new = trans_mat_new.T
    trans_mat_new /= np.sum(trans_mat_new, 0)
    trans_mat_new = trans_mat_new.T

    return sparse.csr_matrix(trans_mat_new)


def do_filter(trans_mat, obs_model, last_belief, filtering_count, belief_init_step):

    [node_num, input_len] = obs_model.shape

    AT = sparse.csr_matrix(trans_mat).T
    belief_all = np.zeros([node_num, input_len])
    belief = last_belief

    for i in range(input_len):
        if filtering_count == 0:  # re-initialize the belief
            init_dist = np.ones([node_num, 1]) / node_num
            belief = normalize_belief(init_dist.flatten() * obs_model[:, 0])
            belief_all[:, i] = belief

            filtering_count += 1
        else:
            belief = normalize_belief(AT.dot(belief) * obs_model[:, i])
            belief_all[:, i] = belief

            filtering_count += 1

            if filtering_count == belief_init_step:  # set the filtering count to zero for belief re-initialization
                filtering_count = 0

        return filtering_count, belief_all


def normalize_belief(belief):
    ''' input should be a vector '''

    z = belief.sum()
    belief = belief / z if z != 0 else 0

    return belief


def normalize(img):
    ''' this function is used to normalize an image of numpy array data type '''

    # this normalized the input image with with mean and stand deviation
    c, h, w = img.shape
    mean = np.mean(img.reshape(c, -1), 1)
    std = np.std(img.reshape(c, -1), 1)
    mean = np.expand_dims(np.expand_dims(mean, axis=1), axis=1)
    std = np.expand_dims(np.expand_dims(std, axis=1), axis=1)

    return (img - mean) / std

    # # this normalized the input image with its maximum value and its minimum value, the output is in range of [0, 1]
    # c, h, w = img.shape
    # img_max = np.max(img.reshape(c, -1), 1)
    # img_min = np.min(img.reshape(c, -1), 1)
    # img_max = np.expand_dims(img_max, [1, 2])
    # img_min = np.expand_dims(img_min, [1, 2])
    #
    # return 2 * ((img - img_min) / (img_max - img_min) - 0.5)
