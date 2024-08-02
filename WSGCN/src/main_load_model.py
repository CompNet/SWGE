"""SGCN runner."""

from sgcn import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs
import torch
import os.path

#sauvegarde et recharge le modèle à chaque run

def main():
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)
    trainer = SignedGCNTrainer(args, edges)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    if os.path.exists("savemodel.txt"):
        trainer.model.load_state_dict(torch.load("savemodel.txt"))
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

    torch.save(trainer.model.state_dict(), "savemodel.txt")

if __name__ == "__main__":
    main()
