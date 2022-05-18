import torch

from tqdm import tqdm
from .base import Trainer


class DrugCellTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, epoch_criterion=None):
        super(DrugCellTrainer, self).__init__(model, criterion, optimizer, config, device,
                                              data_loader, valid_data_loader, lr_scheduler)
        self.epoch_criterion = epoch_criterion
        if config.ckpt is not None:
            self._resume_checkpoint(config.ckpt)

    def train(self):
        self.logger.info("Start Training...")
        if self.epoch_criterion is not None:
            max_epoch_loss = 0
            best_epoch_index = -1
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            if log['val_epoch_loss'] >= max_epoch_loss:
                max_epoch_loss = log['val_epoch_loss']
                best_epoch_index = epoch

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

        if self.epoch_criterion is not None:
            self.logger.info('    {:15s}: {}'.format(
                "Best Performance Epoch", str(best_epoch_index)))

    def _concat_output(self, labels, output):
        if not labels.size()[0]:
            labels = output.detach()
        else:
            labels = torch.cat([labels, output.detach()], dim=0)
        return labels

    def _train_epoch(self, epoch):
        self.model.train()
        log = dict()
        self.logger.info('Epoch %d starts!' % epoch)
        if self.epoch_criterion is not None:
            labels_pred, labels_gold = torch.zeros(0, 0).to(
                self.device), torch.zeros(0, 0).to(self.device)
        for batch_idx, batch in enumerate(tqdm(self.data_loader)):
            if isinstance(batch, tuple):
                data, label = batch[0].to(
                    self.device), batch[1].to(self.device)
            else:
                data, label = batch.to(self.device)
            self.optimizer.zero_grad()
            output, _ = self.model(data)

            if self.epoch_criterion is not None:
                labels_pred = self._concat_output(
                    labels_pred, output['final'].detach())
                labels_gold = self._concat_output(labels_gold, label)

            loss = self.criterion(output, label)
            loss.backward()
            self.model.update_by_mask()
            self.optimizer.step()

            log.update({str(batch_idx): loss.item()})

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

        if self.epoch_criterion is not None:
            epoch_loss = self.epoch_criterion(labels_pred, labels_gold)
            log.update({'epoch_loss': epoch_loss})

        if self.do_validation:
            self.logger.info('Finished Training, Start Validation...')
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        log = dict()

        if self.epoch_criterion is not None:
            labels_pred, labels_gold = torch.zeros(0, 0).to(
                self.device), torch.zeros(0, 0).to(self.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.valid_data_loader)):
                if isinstance(batch, tuple):
                    data, label = batch[0].to(
                        self.device), batch[1].to(self.device)
                else:
                    data, label = batch.to(self.device)

                output, _ = self.model(data)
                loss = self.criterion(output, label)
                log.update({str(batch_idx): loss.item()})
                if self.epoch_criterion is not None:
                    labels_pred = self._concat_output(
                        labels_pred, output['final'].detach())
                    labels_gold = self._concat_output(labels_gold, label)

        if self.epoch_criterion is not None:
            epoch_loss = self.epoch_criterion(labels_pred, labels_gold)
            log.update({'epoch_loss': epoch_loss})

        return log

    def _save_checkpoint(self, epoch):
        """
            Add custom save method for drugcell, which don't save embedding params.
        """
        model_dict = self.model.state_dict()
        for key in model_dict.keys():
            if key == 'vnn.embedding.weight' or key == 'drug.fingerprint.weight':
                model_dict.pop(key)
        state = {
            'epoch': epoch,
            'state_dict': model_dict,
            'optimizer': self.optimizer.state_dict()
        }
        filename = str(self.checkpoint_dir /
                       'ckpt-epoch{}.pt'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))