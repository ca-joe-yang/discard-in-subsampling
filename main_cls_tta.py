import argparse

from tta.trainer import TrainerTTA
from tta.cfg import get_cfg_tta
from utils.random_helper import set_random_seed

def test_learned_tta(
    trainer: TrainerTTA, 
    cfg_filename: str,
    agg_name: str,
    total_budgets: list[int] = [30, 100, 150],
    search_budgets: list[int] = [3, 10, 15],
) -> None:
    for total_budget, search_budget in zip(total_budgets, search_budgets):
        trainer.train(budget=total_budget, best=False)
        trainer.test(total_budget, agg_name, best=False)

        tta_budget = total_budget // search_budget
        trainer.train(budget=tta_budget, best=True)
        trainer.test_poolsearch(
            agg_name,
            cfg_filename, 
            'Ours-learn', 
            search_budget=search_budget,
            tta_budget=tta_budget,
            learn=True, 
            best=True)

def test_nonlearned_tta(
    trainer: TrainerTTA, 
    cfg_filename: str,
    agg_name: str,
    total_budgets: list[int] = [30],
    search_budgets: list[int] = [2],
) -> None:
    for total_budget, search_budget in zip(total_budgets, search_budgets):
        trainer.test(total_budget, agg_name, best=False)

        tta_budget = total_budget // search_budget
        trainer.test_poolsearch(
            agg_name,
            cfg_filename, 
            'Ours', 
            search_budget=search_budget,
            tta_budget=tta_budget,
            learn=False, 
            best=False)

def main(args: argparse.Namespace) -> None:
    cfg_tta = get_cfg_tta(args.opts)
    print(cfg_tta)
    set_random_seed(args.seed)

    trainer = TrainerTTA(cfg_tta, args)
    trainer.gen_aug_logits()

    trainer.test()

    match args.tta:
        case 'MaxTTA' | 'AugTTA' | 'GPS':
            test_learned_tta(trainer, args.config, args.tta)
        case 'MeanTTA' | 'MaxTTA':
            test_nonlearned_tta(trainer, args.config, args.tta)
        case _:
            raise ValueError(args.tta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TTA train')
    parser.add_argument(
        '--config', type=str, default='configs/tta/cifar100.yaml',
        help='config file')
    parser.add_argument(
        '--tta', type=str, default=None,
        help='TTA method')
    parser.add_argument(
        '--seed', type=int, default=0, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--exp-root', type=str, default='experiments',
        help='Experiments directory')
    parser.add_argument(
        '--pretrained-path', type=str, default=None,
        help='Experiments name')
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
