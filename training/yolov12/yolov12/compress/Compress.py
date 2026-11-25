from compress import GM
from compress import prune
import time
import torch.nn as nn
import torch
from ultralytics.nn.modules import *
import yaml
from ultralytics import YOLO


class PruneHandler():
    def __init__(self, model, compression_ratio, method, cfg_output_path, prune_type='ALL'):
        self.model = model
        self.ckpt = model.ckpt['model']
        self.model.cpu()
        self.cr = compression_ratio
        self.method = method
        self.cfg_output_path = cfg_output_path
        self.model.to('cpu')
        self.remain_index_out = {}
        self.prune_type = prune_type

    def prune(self):
        if self.method == 'GM':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not '21' in name:
                        if isinstance(module, nn.Conv2d):
                            if name in ['6.m.0.0.attn.qk.conv', '8.m.0.0.attn.qk.conv']:
                                half_len = module.weight.size(0) // 2
                                q, k = torch.chunk(module.weight, 2)
                                shared_channels = q + k
                                _, mask = GM.gm_structured(shared_channels, name=None, module_name=name, amount=self.cr, dim=0)
                                
                                mask = torch.cat([mask, mask], dim=0)
                                mask = mask.to(device=module.weight.device, dtype=module.weight.dtype).view(-1, 1, 1, 1)

                                with torch.no_grad():
                                    module.weight.mul_(mask)

                                module.weight_mask = mask
                                
                                
                            else:
                                GM.gm_structured(module, name='weight', module_name=name, amount=self.cr, dim=0)
                                mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                                prune.remove(module, 'weight')


                        if isinstance(module, nn.BatchNorm2d):
                            if name in ['6.m.0.0.attn.qk.bn', '8.m.0.0.attn.qk.bn']:
                                mask = mask.view(mask.size(0))
                            prune.l1_unstructured(module, name='weight', module_name=name, amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', module_name=name, amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not any(
                            name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '21']):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0, module_name=name)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', module_name=name, amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', module_name=name, amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if any(name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8']):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0, module_name=name)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', module_name=name, amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', module_name=name, amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L1':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not '21' in name:
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not any(
                            name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '21']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if any(name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L2':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not '21' in name:
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not any(
                            name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '21']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if any(name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=1-self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')


    def reconstruct(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.training = False

        detect_in_channels = []
        concat = {}
        remain_out_channels = [0, 1, 2]
        for name, module in self.model.model.model.named_modules():
            
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '15', '18']:
                    num = int(name)
                    offset = self.model.model.model[num].conv.weight.shape[0]
                    remain_out_channels = module.recon(remain_out_channels)
                if name in ['15', '18']:
                    concat[name] = [remain_out_channels, offset]

            elif isinstance(module, C3k2):
                if name in ['2', '4']:
                    c3k = False
                else:
                    c3k = True
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels, c3k)
                if name in ['20']:
                    detect_in_channels.append(remain_out_channels)
                elif name in ['4']:
                    concat[name] = [remain_out_channels, offset]

            elif isinstance(module, A2C2f):
                e = 0.5
                if name in ['6', '8']:
                    a2 = True
                    n = 2
                else:
                    a2 = False
                    n = 1
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels, e, a2, n)
                if name in ['14', '17']:
                    detect_in_channels.append(remain_out_channels)
                elif name in ['6', '8', '11']:
                    concat[name] = [remain_out_channels, offset]

            elif isinstance(module, Detect):
                remain_out_channels = module.recon(detect_in_channels)

            elif isinstance(module, Concat):
                if name == '10':
                    concat['6'][0] = [x + concat['8'][1] for x in concat['6'][0]]
                    remain_out_channels = concat['8'][0] + concat['6'][0]
                elif name == '13':
                    concat['4'][0] = [x + concat['11'][1] for x in concat['4'][0]]
                    remain_out_channels = concat['11'][0] + concat['4'][0]
                elif name == '16':
                    concat['11'][0] = [x + concat['15'][1] for x in concat['11'][0]]
                    remain_out_channels = concat['15'][0] + concat['11'][0]
                elif name == '19':
                    concat['8'][0] = [x + concat['18'][1] for x in concat['8'][0]]
                    remain_out_channels = concat['18'][0] + concat['8'][0]

    def model_to_yaml(self):
        from_ = -1
        repeats = 1
        yaml_dict = {}
        yaml_dict["nc"] = 8
        yaml_dict["scales"] = {'prune': [1, 1, 1024]}
        yaml_dict["backbone"] = []
        yaml_dict["head"] = []

        for name, module in self.model.ckpt['model'].model.named_modules():
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '15', '18']:
                    args = [module.conv.out_channels, module.conv.kernel_size[0], module.conv.stride[0], module.conv.dilation[0], module.conv.groups]
                    layer = [from_, repeats, type(module).__name__, args]
                    if name in ['15', '18']:
                        yaml_dict["head"].append(layer)
                    else:
                        yaml_dict["backbone"].append(layer)

            elif isinstance(module, C3k2):
                if name in ['2', '4']:
                    c3k = False
                    e = 0.25
                else:
                    c3k = True
                    e = 0.5               
                args = [module.cv2.conv.out_channels, c3k, e]
                layer = [from_, len(module.m), type(module).__name__, args]
                if name in ['20']:
                    yaml_dict["head"].append(layer)
                else:
                    yaml_dict["backbone"].append(layer)

            elif isinstance(module, A2C2f):
                if name in ['6']:
                    a2 = True
                    area = 4
                elif name in ['8']:
                    a2 = True
                    area = 1
                elif name in ['11', '14', '17']:
                    a2 = False
                    area = -1
                args = [module.cv2.conv.out_channels, a2, area]
                layer = [from_, repeats, type(module).__name__, args]
                if name in ['11', '14', '17']:
                    yaml_dict["head"].append(layer)
                else:
                    yaml_dict["backbone"].append(layer)

            elif isinstance(module, Detect):
                args = [yaml_dict["nc"]]
                layer = [[14, 17, 20], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, Concat):
                args = [1]
                if name == '10':
                    layer = [[from_, 6], repeats, type(module).__name__, args]
                elif name == '13':
                    layer = [[from_, 4], repeats, type(module).__name__, args]
                elif name == '16':
                    layer = [[from_, 11], repeats, type(module).__name__, args]
                elif name == '19':
                    layer = [[from_, 8], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, nn.Upsample):
                args = ['None', module.scale_factor, module.mode]
                layer = [from_, repeats, "nn." + type(module).__name__, args]
                yaml_dict["head"].append(layer)

        yaml_str = yaml.dump(yaml_dict)
        with open(f'{self.cfg_output_path}/best.yaml', "w") as file:
            file.write(yaml_str)

    def compress_yolov12(self):
        print('Pruning...')
        start = time.time()
        self.prune()
        self.reconstruct()
        torch.save(self.model, f'{self.cfg_output_path}/best_prune.pt')
        self.model_to_yaml()
        print('Done')
        print(f'time : {time.time() - start}')
        return YOLO(f'{self.cfg_output_path}/best.yaml').load(f'{self.cfg_output_path}/best_prune.pt')
