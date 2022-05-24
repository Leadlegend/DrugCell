import os
import torch

from tqdm import tqdm
from .base import Trainer


class DrugCellTrainer(Trainer):

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 config,
                 device,
                 data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None,
                 epoch_criterion=None):
        super(DrugCellTrainer,
              self).__init__(model, criterion, optimizer, config, device,
                             data_loader, valid_data_loader, lr_scheduler)
        self.epoch_criterion = epoch_criterion
        self.epoch_loss_name = '%s_correlation' % config.epoch_criterion if epoch_criterion is not None else ''
        if config.ckpt is not None:
            self._resume_checkpoint(config.ckpt)
        self.model._to_(self.device)

    def train(self):
        self.logger.info("Start Training...")
        if self.epoch_criterion is not None:
            max_epoch_loss = 0
            best_epoch_index = -1
        for epoch in range(self.start_epoch, self.epochs + 1):
            save_flag = epoch % self.save_period == 0
            result = self._train_epoch(epoch)
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            if log['val_%s'%self.epoch_loss_name] >= max_epoch_loss:
                max_epoch_loss = log['val_%s'%self.epoch_loss_name]
                best_epoch_index = epoch
                if epoch > self.epochs / 3:
                    save_flag = True

            # print logged informations to the screen
            for key, value in log.items():
                if key.endswith('correlation'):
                    self.logger.info('{:15s}: {}'.format(str(key), value))
                else:
                    self.logger.debug('{:15s}: {}'.format(str(key), value))

            if save_flag:
                self._save_checkpoint(epoch)

        if self.epoch_criterion is not None:
            self.logger.info('{:15s}: {}'.format("Best Performance Epoch",
                                                 str(best_epoch_index)))

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
                data, label = batch[0].to(self.device), batch[1].to(
                    self.device)
            else:
                data, label = batch.to(self.device)
            self.optimizer.zero_grad()
            output, _ = self.model(data)

            if self.epoch_criterion is not None:
                labels_pred = self._concat_output(labels_pred,
                                                  output['final'].detach())
                labels_gold = self._concat_output(labels_gold, label)

            loss = self.criterion(output, label)
            loss.backward()
            self.model.update_by_mask()
            self.optimizer.step()

            log.update({str(batch_idx): loss.item()})

            if batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} {}\t|\tLR: {:.5f}\t|\tLoss: {:.6f}'.format(
                        epoch, self._progress(batch_idx),
                        self.optimizer.state_dict()['param_groups'][0]['lr'],
                        loss.item()))
                if self.lr_scheduler is not None:
                    metrics = loss
                    self.lr_scheduler.step(metrics)

        if self.epoch_criterion is not None:
            epoch_loss = self.epoch_criterion(labels_pred, labels_gold)
            log.update({self.epoch_loss_name: epoch_loss})

        if self.do_validation:
            self.logger.info('Finished Training, Start Validation...')
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

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
                    data, label = batch[0].to(self.device), batch[1].to(
                        self.device)
                else:
                    data, label = batch.to(self.device)

                output, _ = self.model(data)
                loss = self.criterion(output, label)
                log.update({str(batch_idx): loss.item()})
                if self.epoch_criterion is not None:
                    labels_pred = self._concat_output(labels_pred,
                                                      output['final'])
                    labels_gold = self._concat_output(labels_gold, label)

        if self.epoch_criterion is not None:
            epoch_loss = self.epoch_criterion(labels_pred, labels_gold)
            log.update({self.epoch_loss_name: epoch_loss})

        return log

    def _save_checkpoint(self, epoch):
        """
            Add custom save method for drugcell, which don't save embedding params.
        """
        model_dict = self.model.state_dict()
        keys = list(model_dict.keys())
        for key in keys:
            if key == 'vnn.embedding.weight' or key == 'drug.fingerprint.weight':
                model_dict.pop(key)
        state = {
            'epoch': epoch,
            'state_dict': model_dict,
            'optimizer': self.optimizer.state_dict()
        }
        if self.lr_scheduler is not None:
            state['scheduler'] = self.lr_scheduler.state_dict()
        filename = os.path.join(self.checkpoint_dir,
                                'ckpt-epoch{}.pt'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
