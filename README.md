================License==================

DeepSSInter is freely available for academic or non-commercial users. 
Released under GNU General Public License Version 3.

================About DeepSSInter==================

The program is developed to predict the contact preditions across the interfaces of
the complexes with the two monomer structure as input. 


==============Software Requirements=================

1. the required packages are listed in environment.yaml, install it according the conda commnad
    conda env create -f environment.yaml

2. ESM2 pre-trained model

    You may need to download the pretrained model [esm2_t33_650M_UR50D.pt](https://github.com/facebookresearch/esm) 
    and the regression model [esm2_t33_650M_UR50D-contact-regression.pt] 
    (https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt).

    You should put two pretrained models in the same directory "DeepSSInter/project/utils/ems2"

3. Saprot pre-trained model

    You may need to download the pretrained model [SaProt_650M_AF2.pt](https://huggingface.co/westlake-repl/SaProt_650M_AF2)

    You should put two pretrained models in the same directory "DeepSSInter/project/utils/saprot"

4. Install with DeepSSInter with pip

    pip install -e .


=================Examples=================

Here is a demo to run DeepSSInter:

    cd project

    python lit_model_predict.py --left_pdb_filepath examples/5FGL_A.pdb --right_pdb_filepath examples/5FGL_B.pdb --ckpt_dir model/ --ckpt_name model_checkpoint.ckpt

    The predicted residue-residue contacts are saved in examples/5FGL_contact_prob_map.npy


