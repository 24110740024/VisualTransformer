# Reference syntax-tree fixtures for the reported v6 implementation.
# Parsed by tests; not executable entrypoints.

def create_model(pretrained=False):
    model = vit_b_16(weights=None, dropout=0.0, attention_dropout=0.0)
    if pretrained:
        verify_weights()
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu', weights_only=True), strict=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    require(model.image_size == 224 and model.patch_size == 16 and (model.hidden_dim == 768), 'Unexpected torchvision model geometry')
    require(len(model.encoder.layers) == 12 and model.encoder.ln.eps == 1e-06, 'Unexpected torchvision encoder defaults')
    require(all((p.requires_grad for p in model.parameters())), 'Candidate must finetune all parameters')
    return model

def build_optimizer(model, backbone_lr=1e-05, head_lr=0.0001, weight_decay=0.05):
    head = list(model.heads.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    require(head and backbone and (len(head_ids) == len(head)), 'Invalid optimizer parameter groups')
    require({id(p) for p in backbone}.isdisjoint(head_ids), 'Optimizer groups overlap')
    require(len(backbone) + len(head) == len(list(model.parameters())), 'Optimizer omitted parameters')
    return torch.optim.AdamW([{'params': backbone, 'lr': backbone_lr, 'group_name': 'backbone'}, {'params': head, 'lr': head_lr, 'group_name': 'head'}], weight_decay=weight_decay)
