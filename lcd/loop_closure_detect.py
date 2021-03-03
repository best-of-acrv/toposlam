from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import faiss
import numpy as np
from scipy import sparse

from lcd.hmm_utils import do_filter, graph_update, normalize
from lcd.extract_deep_vlad_feature import ExtractDeepVladFeature


class LoopClosureDetect(nn.Module):
    def __init__(self, device, net_vlad_ckp, belief_thr=0.15, candidate_num=5, frame_step_thr=200, belief_init_step=10):
        super(LoopClosureDetect, self).__init__()

        ''' Parameters '''
        # HMM parameters
        self.hmm_params = {}
        self.hmm_params["W"] = 10  # the maximum time step size to construct graph edge, frame interval larger than
        # this value will result in transition probability of zero
        self.hmm_params["distance_upper_bound"] = 2.0
        self.hmm_params["large_distance"] = 2.5
        self.hmm_params["sigma"] = 0.3
        self.hmm_params['belief_thr'] = belief_thr  # belief larger than this value will be accepted as a successful
        # match
        self.hmm_params['candidate_num'] = candidate_num  # the number of the potential candidate from which to
        # choose the loop closure detection result
        self.hmm_params['frame_step_thr'] = frame_step_thr  # only the frame far away from the query frame than this
        # value is accepted as a successful match, this is used to avid matching to the frame close to the query frame
        self.hmm_params['belief_init_step'] = belief_init_step  # the belief will be re-initialized every this
        # number of frames

        # FAISS knn search parameters
        self.faiss_params = {}
        self.faiss_params['knn'] = 5
        self.faiss_params['feature_dim'] = 32768  # 32768 is the dimension of the feature produced by NetVlad

        # graph related variables, their sizes are updated within each call of the forward() function
        self.image_ids = []
        self.trans_mat = sparse.csr_matrix(np.zeros([0, 0]))
        self.last_belief = np.ones([0])

        # device
        self.device = device

        # the belief will be re-initialized if this value is larger than self.hmm_params['belief_init_step']
        self.filtering_count = 0

        # if more than self.hmm_params['belief_init_step'] input frames do not perform loop closure detection, we also
        # re-initialize the belief
        self.no_detect_count = 0

        ''' functions '''
        # NetVlad for deep vlad feature extraction
        self.extract_deep_vlad_feature = ExtractDeepVladFeature(net_vlad_ckp)
        self.extract_deep_vlad_feature.to(device=self.device)

        # FAISS for nearest neighbors search, brute force L2 distance search in FAISS
        self.faiss = faiss.IndexFlatL2(self.faiss_params['feature_dim'])

    def forward(self, image, image_id, do_closure_detect):
        '''
        Input:
               image:        a list containing n images for loop closure detection, dtype: numpy array
               image_id:     a list of length n, containing the ids of the input images, dtype: int

        Output:
               matched_pairs:     loop closure detection results, a list containing n pairs of
                                  [input_id, matched_id, belief]
                                  return [None, None, None] if there are no successful matches
        '''

        assert len(image) == len(image_id), 'the number of input image should be the same as that of image ids'

        # extract deep vlad features of input images
        input_len = len(image)
        c, h, w = image[0].shape  # input image of size c x h x w
        tensor_image = torch.zeros([input_len, c, h, w], requires_grad=False)

        # normalize the input
        for i in range(input_len):
            image_norm = normalize(image[i].astype('float64'))
            tensor_image[i, ...] = torch.from_numpy(image_norm)

        tensor_image = tensor_image.to(device=self.device)
        vlad_feature = self.extract_deep_vlad_feature(tensor_image)
        vlad_feature = vlad_feature.detach().cpu().numpy()  # dtype: float32

        node_num = self.faiss.ntotal  # total number of previously input images (the 'database' for knn search),
        # i.e. the number of the nodes in the graph

        # perform loop closure detection if 'do_closure_detect' is true
        if do_closure_detect:

            if node_num <= self.hmm_params["W"] or node_num <= self.faiss_params['knn']:
                # we perform loop closure detection only when the number of images in the
                # database is larger than the maximum time step or the number of knn neighbors

                matched_pairs = []
                for i in range(input_len):
                    matched_pairs.append([None, None, None])  # no matching results, return None

            else:  # find nearest neighbors of the input images

                nearest_distances, nearest_indices \
                    = self.faiss.search(vlad_feature, self.faiss_params['knn'])
                nearest_distances = nearest_distances.astype('float64')

                # Create observation model from K-NN
                obs_model = np.ones([node_num, input_len]) * self.hmm_params["large_distance"]
                obs_model = np.exp(- obs_model / self.hmm_params["sigma"])

                nearest_distances[nearest_distances > self.hmm_params["distance_upper_bound"]] \
                    = self.hmm_params["large_distance"]

                nearest_probs = np.exp(- nearest_distances / self.hmm_params["sigma"])
                for i in range(input_len):
                    obs_model[nearest_indices[i, :], i] = nearest_probs[i, :]

                # Perform Bayes filtering
                self.filtering_count, belief_all = do_filter(self.trans_mat, obs_model, self.last_belief,
                                                             self.filtering_count, self.hmm_params['belief_init_step'])

                # Find matches of the input images in the database
                matched_pairs = []

                for i in range(input_len):
                    belief = belief_all[:, i]
                    local_match = []
                    candidate_idx = (-belief).argsort()[:self.hmm_params['candidate_num']]
                    # find the first 'self.hmm_params['candidate_num']' items in the belief

                    for j in range(len(candidate_idx)):
                        # close to the query frame, not a successful match, continue
                        if node_num - candidate_idx[j] + i < self.hmm_params['frame_step_thr']:
                            continue

                        # only the candidate far away from the query image and its belief larger than the threshold is a
                        # successful match
                        if belief[candidate_idx[j]] > self.hmm_params['belief_thr']:
                            local_match = [image_id[i], self.image_ids[candidate_idx[j]], belief[candidate_idx[j]]]
                            break  # have found the matched frame with the largest belief, thus break

                    if len(local_match) == 0:  # no successful match
                        matched_pairs.append([None, None, None])
                    else:
                        matched_pairs.append(local_match)

                # extend the last belief to have the same length as the self.database.shape[0]
                self.last_belief = belief_all[:, -1]
                self.last_belief = np.concatenate([self.last_belief, np.zeros(input_len)])

        else:  # do not perform loop closure detection
            matched_pairs = []
            for i in range(input_len):
                matched_pairs.append([None, None, None])

            self.no_detect_count += input_len
            if self.no_detect_count >= self.hmm_params['belief_init_step']:  # re-initialize the belief if many input
                # frames do not perform loop closure detection
                self.last_belief = np.ones(node_num + input_len) / (node_num + input_len)
                self.no_detect_count = 0
            else:
                self.last_belief = np.concatenate([self.last_belief, np.zeros(input_len)])

        #  update the graph
        self.image_ids, self.trans_mat\
            = graph_update(node_num, self.image_ids, self.trans_mat, vlad_feature, image_id, self.hmm_params['W'])

        # adding the features of the input frames to the faiss index structure
        self.faiss.add(vlad_feature)

        # return loop closure detection results
        return matched_pairs
