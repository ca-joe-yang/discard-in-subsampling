import torch

class BasePolicy:

    def budget2augs(self, budget):
        names = self.transforms_names[:budget]
        return names

    def get_transforms(self, start_idx=0, end_idx=-1):
        fns = self.fns[start_idx:end_idx]
        return fns

    def name2transtorms(self, names):
        return [ self.name2fns[name] for name in names ]

    def get_best_transforms(self, agg_model, budget):
        fns = []
        match agg_model.name:
            case 'AugTTA':
                coeffs = agg_model.coeffs
                idxs = torch.topk(coeffs, k=budget, dim=0).indices[:, 0].cpu().numpy()
            case 'ClassTTA':
                coeffs = agg_model.coeffs.mean(dim=1, keepdims=True) #A, K
                idxs = torch.topk(coeffs, k=budget, dim=0).indices[:, 0].cpu().numpy()
            case 'GPS':
                idxs = agg_model.idxs
            case _:
                raise ValueError(agg_model.name)

        fns = [self.fns[i] for i in idxs ]
        return fns, idxs
    