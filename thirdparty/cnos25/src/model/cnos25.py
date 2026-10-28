# Derived from https://github.com/Vision-Kek/cnos25 (commit d92e3d3, MIT License; see thirdparty/cnos25/LICENSE).
# Modified by POSE-Lab, National Technical University of Athens (NovaSplat), 2026.
import torch
import logging
import os
import os.path as osp

from src.utils.detection_utils import Detections
from src.model import matching


class CNOS25:
    def __init__(self, proposal, descriptor, matching_metric, device):
        logging.info('Initializing CNOS25')
        self.device = device
        # Init proposal
        self.proposal = proposal
        # Init descriptor
        self.descriptor = descriptor
        self.feat_type = descriptor.cfg.token_type
        self.reference_descriptors = self.load_reference_descriptors()  # the templates
        # Init matching
        self.matching_strategy = matching_metric
        logging.info('Initializing CNOS25 Done.')

    def load_reference_descriptors(self):
        descriptor_pth = osp.join(self.descriptor.cfg.cache_dir, self.descriptor.cfg.cache_file)
        if not osp.isdir(self.descriptor.cfg.cache_dir):
            raise FileNotFoundError(
                f"Descriptor cache directory does not exist: {self.descriptor.cfg.cache_dir}"
            )
        if not osp.isfile(descriptor_pth):
            raise FileNotFoundError(
                f"Descriptor cache file does not exist: {descriptor_pth}"
            )
        logging.info(
            "Available descriptors %s selected %s.",
            os.listdir(self.descriptor.cfg.cache_dir),
            self.descriptor.cfg.cache_file,
        )
        logging.info("Loading descriptors from %s...", descriptor_pth)

        dino_descriptors = torch.load(descriptor_pth, map_location=torch.device('cpu'), weights_only=False)
        if 'template_embeds' in dino_descriptors: # support this as well as flat dict
            logging.info('found template_embeds key')
            dino_descriptors = dino_descriptors['template_embeds']
        if not isinstance(dino_descriptors, dict):
            raise TypeError(
                f"Expected descriptor cache to be a dict, got {type(dino_descriptors)!r}"
            )

        # select the feature type (templates) shape:(n,k,d)
        matching_keys = [k for k in dino_descriptors.keys() if k.startswith(self.feat_type)]
        if len(matching_keys) == 0:
            available_keys = sorted(dino_descriptors.keys())
            raise ValueError(
                f"No descriptor keys start with '{self.feat_type}'. Available keys: {available_keys}"
            )

        dino_descriptors = torch.stack([dino_descriptors[k] for k in matching_keys])
        logging.info('Loaded %s, %s.', self.feat_type, dino_descriptors.shape)
        if dino_descriptors.shape[-1] == 2048:
            logging.info(
                'Found %s, likely because of dino `torch.cat(features, dim=-1)`. Using only [:1024]',
                dino_descriptors.shape,
            )
            dino_descriptors = dino_descriptors[...,:1024]

        return dino_descriptors.to(self.device)

    def proposal_fwd_pass(self, pil_image):
        proposals = self.proposal.model.predict_proposals(pil_image)
        proposals = Detections({'masks': proposals.mask, 'boxes': proposals.xyxy})
        # keep only max top-n
        proposals.retain_top_n_confident_detections(self.proposal.model.cfg.keep_n_top_confident_proposals)
        return proposals

    def dino_fwd_pass(self, pil_image, proposals):
        if len(proposals) == 0:
            logging.info('Empty proposals. Skipping.')
            return torch.zeros(0,self.reference_descriptors.shape[-1], device=self.device)

        proposal_img_tensors = self.descriptor.model.process_rgb_proposals(pil_image, proposals.masks, proposals.boxes)

        assert self.feat_type in ['x_norm_clstoken','etc...']
        proposal_descriptors = self.descriptor.model.chunked_fwd(proposal_img_tensors)[self.feat_type]
        return proposal_descriptors

    def classify(self, proposal_descriptors):
        if len(proposal_descriptors) == 0:
            return torch.tensor([]), torch.tensor([])

        score_per_proposal, assigned_idx_object = matching.classify_proposals_through_matching(
            object_proposal_class_embeddings=proposal_descriptors,
            ref_samples=self.reference_descriptors,
            matching_strategy=self.matching_strategy)
        return score_per_proposal, assigned_idx_object