import torch


def transfer(path, new_path='./model.pt', if_model=False):
    """
        transfer the ckpt from author to ckpt that
    """
    if not if_model:
        sd_old = torch.load(path)
        keys = list(sd_old.keys())
        for key in keys:
            if key.startswith('final'):
                continue
            elif key.startswith('drug'):
                sd_old['drug.' + key] = sd_old.pop(key)
            else:
                sd_old['vnn.' + key] = sd_old.pop(key)
        torch.save(sd_old, new_path)
    else:
        print("please transfer the checkpoint into state_dict first.")


if __name__ == '__main__':
    path = './ckpt/dc_v0.pt'
    transfer(path)
