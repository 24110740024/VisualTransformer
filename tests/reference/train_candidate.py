# Reference syntax-tree fixtures for the reported v6 implementation.
# Parsed by tests; not executable entrypoints.

def main(argv=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generators = {key: torch.Generator().manual_seed(args.seed + offset) for offset, key in enumerate(('sampling', 'train_workers', 'val_workers'))}
    train_set, val_set = (ManifestDataset(train_rows, True), ManifestDataset(val_rows, False))
    counts = np.bincount([row['label'] for row in train_rows], minlength=2)
    if args.weighted_sampler:
        weights = sampling_weights
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generators['sampling'])
    else:
        sampler = RandomSampler(train_set, generator=generators['sampling'])
    loader_options = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, persistent_workers=False)
    train_loader = DataLoader(train_set, sampler=sampler, generator=generators['train_workers'], **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, generator=generators['val_workers'], **loader_options)
    model = create_model(pretrained=True).to(device)
    if args.class_weight:
        raise ValueError('v6 requires unweighted cross entropy')
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start_epoch, best_auc, best_epoch, no_improve, history = (1, -1.0, -1, 0, [])
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc, best_epoch = (checkpoint['best_auc'], checkpoint['best_epoch'])
        no_improve, history = (checkpoint['no_improve'], checkpoint['history'])
        restore_rng(checkpoint['rng'], generators)
        if best_epoch == checkpoint['epoch']:
            atomic_torch_save(checkpoint, out_dir / 'best.pth')
        else:
            previous_best = torch.load(out_dir / 'best.pth', map_location='cpu', weights_only=False)
            if previous_best['config_sha256'] != config_hash or previous_best['epoch'] != best_epoch:
                raise ValueError('Selected best checkpoint differs from last committed epoch')
        atomic_json({'best_epoch': best_epoch, 'best_val_auc_lesion': best_auc, 'completed_epoch': checkpoint['epoch'], 'no_improve': no_improve, 'config_sha256': config_hash}, out_dir / 'status.json')
        write_history(history, out_dir / 'history.csv')
        if no_improve >= args.patience:
            print('This run already stopped for patience; no additional training performed.')
            return
    print(json.dumps({'event': 'training_started', 'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None, 'seed': args.seed, 'train_records': len(train_set), 'val_records': len(val_set), 'batches_per_epoch': len(train_loader), 'start_epoch': start_epoch, 'config_sha256': config_hash}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started, lr_used = (time.perf_counter(), optimizer.param_groups[0]['lr'])
        head_lr_used = optimizer.param_groups[1]['lr']
        model.train()
        train_loss, train_correct, train_total = (0.0, 0, 0)
        for batch_index, (image, label) in enumerate(train_loader, 1):
            image, label = (image.to(device, non_blocking=True), label.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(image)
                loss = criterion(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at epoch {epoch}; last completed epoch remains resumable')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * label.size(0)
            train_correct += (logits.argmax(1) == label).sum().item()
            train_total += label.size(0)
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(train_loader):
                progress = {'event': 'batch_progress', 'phase': 'train', 'epoch': epoch, 'batch': batch_index, 'batches': len(train_loader), 'elapsed_seconds': time.perf_counter() - started, 'train_loss_so_far': train_loss / train_total}
                atomic_json(progress, out_dir / 'live.json')
                print(json.dumps(progress), flush=True)
        scheduler.step()
        model.eval()
        val_loss, val_correct, val_total, probabilities, targets = (0.0, 0, 0, [], [])
        with torch.no_grad():
            for image, label in val_loader:
                image, label = (image.to(device, non_blocking=True), label.to(device, non_blocking=True))
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    logits = model(image)
                    loss = criterion(logits, label)
                val_loss += loss.item() * label.size(0)
                val_correct += (logits.argmax(1) == label).sum().item()
                val_total += label.size(0)
                probabilities.extend(logits.softmax(1)[:, 0].cpu().tolist())
                targets.extend((label == 0).long().cpu().tolist())
        val_auc = float(roc_auc_score(targets, probabilities))
        if not np.isfinite(val_auc) or not np.isfinite(val_loss):
            raise RuntimeError('Nonfinite validation metrics; last completed epoch remains resumable')
        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch, no_improve = (val_auc, epoch, 0)
        else:
            no_improve += 1
        history.append(dict(epoch=epoch, train_loss=train_loss / train_total, train_acc=train_correct / train_total, val_loss=val_loss / val_total, val_acc=val_correct / val_total, val_auc_lesion=val_auc, lr_used=lr_used, head_lr_used=head_lr_used, seconds=time.perf_counter() - started, improved=improved))
        state = {'format_version': FORMAT_VERSION, 'config_sha256': config_hash, 'configuration': configuration, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best_auc': best_auc, 'best_epoch': best_epoch, 'no_improve': no_improve, 'history': history, 'rng': rng_state(generators), 'classes': CLASSES, 'positive_class_index': 0, 'initialization': INITIALIZATION}
        atomic_torch_save(state, out_dir / 'last.pth')
        if improved:
            atomic_torch_save(state, out_dir / 'best.pth')
        write_history(history, out_dir / 'history.csv')
        atomic_json({'best_epoch': best_epoch, 'best_val_auc_lesion': best_auc, 'completed_epoch': epoch, 'no_improve': no_improve, 'config_sha256': config_hash}, out_dir / 'status.json')
        print(json.dumps(history[-1]), flush=True)
        if no_improve >= args.patience:
            print(f'Early stopping after epoch {epoch}; best validation AUC epoch {best_epoch}', flush=True)
            break
