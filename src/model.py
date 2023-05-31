import torch
import lightning.pytorch as pl


class BPModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def step(self, batch, batch_idx):
        x, x_mask, y = batch
        output = self.model(x, x_mask)
        return output, y

    def training_step(self, batch, batch_idx):
        output, y = self.step(batch, batch_idx)
        loss = torch.nn.functional.mse_loss(output, y)
        self.log("Average Train Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, y = self.step(batch, batch_idx)
        sbp_loss, dbp_loss, loss = self.calculate_mse(output, y)
        self.log("Val Loss", loss)
        self.log("sbp_loss", sbp_loss)
        self.log("dbp_loss", dbp_loss)
        return loss

    def test_step(self, batch, batch_idx):
        output, y = self.step(batch, batch_idx)
        sbp_loss, dbp_loss, loss = self.calculate_mse(output, y)
        sbp_mae, dbp_mae, mae = self.calculate_mae(output, y)

        accuracy = self.calculate_accuracy(output, y)

        self.log("sbp_loss", sbp_loss)
        self.log("dbp_loss", dbp_loss)
        self.log("Test Loss", loss)

        self.log("sbp_MAE", sbp_mae)
        self.log("dbp_MAE", dbp_mae)
        self.log("MAE", mae)

        for target, results in zip([5, 10, 15], accuracy):
            self.log(f"accuracy_{target}", results)

        return {
            "sbp_loss": torch.sqrt(sbp_loss),
            "dbp_loss": torch.sqrt(dbp_loss),
            "overall_loss": loss,
        }

    def calculate_mse(self, output, y):
        sbp_loss = torch.nn.functional.mse_loss(output[-1, 0], y[-1, 0])
        dbp_loss = torch.nn.functional.mse_loss(output[-1, 1], y[-1, 1])
        loss = torch.nn.functional.mse_loss(output, y)
        return sbp_loss, dbp_loss, loss

    def calculate_mae(self, output, y):
        sbp_loss = torch.nn.functional.l1_loss(output[-1, 0], y[-1, 0])
        dbp_loss = torch.nn.functional.l1_loss(output[-1, 1], y[-1, 1])
        loss = torch.nn.functional.l1_loss(output, y)
        return sbp_loss, dbp_loss, loss

    def calculate_loss(self, output, y):
        return self.calculate_mse(output, y)

    def calculate_abp(self, output):
        sbp = output[-1, 0]
        dbp = output[-1, 1]
        return dbp + (0.3 * (sbp - dbp))

    def calculate_accuracy(self, output, y):
        """
        Estimates "accuracy" of reading based on the
        proportion of predictions having less than 5,10,15
        absolute error.
        """
        abp_estimate = self.calculate_abp(output)
        abp_gt = self.calculate_abp(y)

        propto_5 = self._calculate_accuracy(abp_estimate, abp_gt, 5)
        propto_10 = self._calculate_accuracy(abp_estimate, abp_gt, 10)
        propto_15 = self._calculate_accuracy(abp_estimate, abp_gt, 15)
        return propto_5, propto_10, propto_15

    def _calculate_accuracy(self, abp_estimate, abp_gt, target):
        delta = torch.abs(abp_estimate - abp_gt)
        hits = torch.where(delta < target, 1, 0).double()
        return torch.mean(hits)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
