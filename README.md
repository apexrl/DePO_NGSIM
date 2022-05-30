# DePO codes on NGSIM driving experiments

Official Pytorch implemetation of ICML2022 paper [Depo (Plan Your Target and Learn Your Skills: Transferable State-Only Imitation Learning via Decoupled Policy Optimization)](https://arxiv.org/abs/2203.02214) on NGSIM driving dataset. Since some settings of the driving environment are quite different from the Mujoco environments, we maintain the experiment code on NGSIM separately in this repo. The code for the main experiments can be found in [DePO](https://github.com/apexrl/DePO). Current simulation platform is a (slightly) modified version of [PPUU](https://github.com/Atcold/pytorch-PPUU). We plan to open source the training code based on the [SMARTS](https://github.com/huawei-noah/SMARTS) simulator, so be sure to stay tuned!

**Important Notes**

This repository is based on [ILSwiss](https://github.com/Ericonaldo/ILSwiss).

## Reproducing Results

### Download and Transform Driving Dataset

Go to this [address](http://bit.ly/PPUU-data) and download the TGZ file (330 MB) on your machine. Place the downloaded folder as `pytorch-PPUU/traffic-data`.

Then transform the dataset so that it can fit in ILSwiss.

```bash
cd ILSwiss/
python run_experiment.py -e exp_specs/expert_ppuu.yaml
python run_experiment.py -e exp_specs/expert_ppuu_multitype.yaml
```

### Example scripts

#### Co-training of DePO on multiple action spaces

```bash
python run_experiment.py -e exp_specs/dpo_ppuu_multitype.yaml
```

#### Transferring of DePO

1. Pretrain on Normal

```bash
python run_experiment.py -e exp_specs/dpo_ppuu_pretrain.yaml
```

2. Transfer to Inverse and Transpose

Remember to change `state_predictor_path` in `exp_specs/dpo_ppuu_transfer.yaml` to the path of `best.pkl` file obtained from pretraining.

```bash
python run_experiment.py -e exp_specs/dpo_ppuu_transfer.yaml
```

#### Baseline Algorithm

Run GAIfO.

```bash
python run_experiment.py -e exp_specs/gailfo_ppuu.yaml
```
