import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dgl.data import DGLDataset
from torch.utils.data import DataLoader

from project.utils.deepinteract_constants import NODE_COUNT_LIMIT, RESIDUE_COUNT_LIMIT
from project.utils.gt_esm_saprot_res_tri_rbf_modules import LitGINI
from project.utils.deepssinter_utils import collect_args, process_args, custom_dgl_picp_collate_predict
from project.utils.gen_prediction_feats import get_features

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------

class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):

        state_dict = { k:v for k, v in pl_module.state_dict().items() if not k.startswith('esm2') }
        checkpoint['state_dict'] = state_dict

        return checkpoint


class InputDataset(DGLDataset):
    r"""A temporary Dataset for processing and presenting complex data for prediction.

    Parameters
    ----------
    left_pdb_filepath: str
        A filepath to the left input PDB chain. Default: 'test_data/4heq_l_u.pdb'.
    right_pdb_filepath: str
        A filepath to the right input PDB chain. Default: 'test_data/4heq_r_u.pdb'.
    knn: int
        How many nearest neighbors to which to connect a given node. Default: 20.
    force_reload: bool
        Whether to reload the dataset. Default: False.
    verbose: bool
        Whether to print out progress information. Default: False.

    """

    def __init__(self,
                 left_pdb_filepath=os.path.join('test_data', '4heq_l_u.pdb'),
                 right_pdb_filepath=os.path.join('test_data', '4heq_r_u.pdb'),
                 knn=20,
                 force_reload=False,
                 verbose=False):
        assert os.path.exists(left_pdb_filepath), f'Left PDB file not found: {left_pdb_filepath}'
        assert os.path.exists(right_pdb_filepath), f'Right PDB file not found: {right_pdb_filepath}'
        self.left_pdb_filepath = left_pdb_filepath
        self.right_pdb_filepath = right_pdb_filepath
        self.knn = knn
        self.data = {}

        raw_dir = os.path.join(*left_pdb_filepath.split(os.sep)[:-1])
        super(InputDataset, self).__init__(name='InputDataset',
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)
        logging.info(f"Loading complex for prediction,"
                     f" l_chain: {self.left_pdb_filepath}, r_chain: {self.right_pdb_filepath}")

    def download(self):
        """Download an input complex."""
        pass

    def process(self):
        """Process each protein complex into a prediction-ready dictionary representing both chains."""
        # Process the unprocessed protein complex
        left_complex_graph, left_dist, left_seq, left_struct_seq = get_features(self.left_pdb_filepath)
        right_complex_graph, right_dist, right_seq, right_struct_seq = get_features(self.right_pdb_filepath)
        self.data = {
            'graph1': left_complex_graph,
            'graph2': right_complex_graph,
            'seqA': left_seq,
            'seqB': right_seq,
            'struct_seqA': left_struct_seq,
            'struct_seqB': right_struct_seq,
            'distA': left_dist,
            'distB': right_dist,
        }

    def has_cache(self):
        """Check if the input complex is available for prediction."""
        pass

    def __getitem__(self, _):
        """Return requested complex to DataLoader."""
        return self.data

    def __len__(self) -> int:
        """Number of complexes in the dataset."""
        return 1

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each graph node."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        return 28

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 27

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir


def main(args):
    # -----------
    # Input
    # -----------
    input_dataset = InputDataset(left_pdb_filepath=args.left_pdb_filepath,
                                 right_pdb_filepath=args.right_pdb_filepath,
                                 knn=20)
    input_dataloader = DataLoader(input_dataset, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=custom_dgl_picp_collate_predict)

    # -----------
    # Model
    # -----------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)

    model = LitGINI(num_node_input_feats=input_dataset.num_node_features,
                    num_edge_input_feats=input_dataset.num_edge_features,
                    gnn_activ_fn=nn.SiLU(),
                    num_classes=input_dataset.num_classes,
                    max_num_graph_nodes=NODE_COUNT_LIMIT,
                    max_num_residues=RESIDUE_COUNT_LIMIT,
                    testing_with_casp_capri=dict_args['testing_with_casp_capri'],
                    training_with_db5=dict_args['training_with_db5'],
                    pos_prob_threshold=0.5,
                    gnn_layer_type=dict_args['gnn_layer_type'],
                    num_gnn_layers=dict_args['num_gnn_layers'],
                    num_gnn_hidden_channels=dict_args['num_gnn_hidden_channels'],
                    num_gnn_attention_heads=dict_args['num_gnn_attention_heads'],
                    knn=dict_args['knn'],
                    interact_module_type=dict_args['interact_module_type'],
                    num_interact_layers=dict_args['num_interact_layers'],
                    num_interact_hidden_channels=dict_args['num_interact_hidden_channels'],
                    use_interact_attention=dict_args['use_interact_attention'],
                    num_interact_attention_heads=dict_args['num_interact_attention_heads'],
                    disable_geometric_mode=dict_args['disable_geometric_mode'],
                    num_epochs=dict_args['num_epochs'],
                    pn_ratio=dict_args['pn_ratio'],
                    dropout_rate=dict_args['dropout_rate'],
                    metric_to_track=dict_args['metric_to_track'],
                    weight_decay=dict_args['weight_decay'],
                    batch_size=1,
                    lr=dict_args['lr'],
                    pad=dict_args['pad'],
                    viz_every_n_epochs=dict_args['viz_every_n_epochs'],
                    use_wandb_logger=False,
                    weight_classes=dict_args['weight_classes'],
                    fine_tune=False,
                    ckpt_path=None)
    args.experiment_name = f'LitGINI-b{args.batch_size}-gl{args.num_gnn_layers}' \
                           f'-n{args.num_gnn_hidden_channels}' \
                           f'-e{args.num_gnn_hidden_channels}' \
                           f'-il{args.num_interact_layers}-i{args.num_interact_hidden_channels}' \
        if not args.experiment_name \
        else args.experiment_name


    # ------------
    # Checkpoint
    # ------------
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt_provided = args.ckpt_name != ''
    assert ckpt_provided and os.path.exists(ckpt_path), 'A valid checkpoint filepath must be provided'
    checkpoint = torch.load(ckpt_path)
    model_dict = model.state_dict()
    model_dict.update(checkpoint['state_dict'])
    model.load_state_dict(model_dict)
    model.freeze()

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # -----------
    # Prediction
    # -----------
    # Predict with a trained model using the provided input data module
    predict_payload = trainer.predict(model=model, dataloaders=input_dataloader)[0]

    logits = predict_payload[0]
    contact_prob_map =  logits[0].squeeze().cpu().numpy()

    # -----------
    # Saving
    # -----------
    pdb_code = args.left_pdb_filepath.split(os.sep)[-1].split('_')[0]
    input_prefix = os.path.join(*args.left_pdb_filepath.split(os.sep)[:-1])
    contact_map_filepath = os.path.join(input_prefix, f'{pdb_code}_contact_prob_map.npy')
    np.save(contact_map_filepath, contact_prob_map)
    logging.info(f'Saved predicted contact probability map for {pdb_code} as {contact_map_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # -----------
    # Arguments
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Let the model add what it wants
    parser = LitGINI.add_model_specific_args(parser)

    # Re-parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = 'dp'  # Predict using Data Parallel (DP) and not Distributed Data Parallel (DDP) to avoid errors
    args.auto_select_gpus = args.auto_choose_gpus
    args.gpus = args.num_gpus  # Allow user to choose how many GPUs they would like to use for inference
    args.num_nodes = 1  # Enforce predictions to take place on a single node
    args.precision = args.gpu_precision
    args.accumulate_grad_batches = args.accum_grad_batches
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo
    args.stochastic_weight_avg = args.stc_weight_avg

    # Finalize all arguments as necessary
    args = process_args(args)

    # Begin execution of model training with given args
    main(args)
