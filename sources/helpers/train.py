import os
import torch.nn as nn
import mlflow
import torch
from losses import DiceLoss
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large,DeepLabV3_MobileNet_V3_Large_Weights,deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
import torch.optim as optim

from sources.helpers.logger import LoggerHelper
from sys import platform

class TrainingHelper:
    def __init__(self, model_name,num_classes,num_epochs,train_data_generator,val_data_generator,loss_function,optimizer):
        self.logger = LoggerHelper(logger_name="TrainingHelper").logger
        self.device = torch.device("mps" if torch.backends.mps.is_built() else "cpu") if platform == "darwin" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.train_data_generator = train_data_generator
        self.val_data_generator = val_data_generator
        self.model_name = model_name
        self.output_model_name = f"{model_name}_{num_epochs}epochs_{loss_function}_{optimizer}"

        match model_name:
            case "deeplabv3_mobilenet_v3_large":
                self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
                self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

            case "deeplabv3_resnet50":
                self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
                self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

            case "deeplabv3_plus_resnet50":
                self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

            case _:
                raise ValueError(f"Model {model_name} not supported!")

        self.model = self.model.to(self.device)

        match loss_function:
            case "cross_entropy":
                # SparseCategoricalCrossEntropyLoss
                # Jaccard Loss
                # Dice Loss
                # CEDiceLoss
                self.criterion = nn.CrossEntropyLoss()
            case "dice_loss":
                self.criterion = DiceLoss()
            case _:
                raise ValueError(f"Loss function {loss_function} not supported!")

        match optimizer:
            case "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
            case _:
                raise ValueError(f"Optimizer {optimizer} not supported!")

    def __calculate_metrics(self, preds, targets):
        """Calculate IoU, F1-score, and accuracy."""
        metrics = {"IoU": [], "F1-Score": [], "Pixel Accuracy": 0.0}

        # Convert predictions to class labels
        preds = torch.argmax(preds, dim=1)

        intersection = torch.zeros(self.num_classes, device=preds.device)
        union = torch.zeros(self.num_classes, device=preds.device)
        TP = torch.zeros(self.num_classes, device=preds.device)  # True Positives
        FP_FN_error = torch.zeros(self.num_classes, device=preds.device)  # False Positives + False Negatives

        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = targets == c

            intersection[c] = torch.sum((pred_c & target_c).float())  # Intersection
            union[c] = torch.sum((pred_c | target_c).float())  # Union
            TP[c] = intersection[c]
            FP_FN_error[c] = torch.sum(pred_c.float()) + torch.sum(target_c.float()) - TP[c]

        # IoU per class
        metrics["IoU"] = (intersection / (union + 1e-7)).cpu().numpy()
        # F1-Score per class
        metrics["F1-Score"] = (2 * TP / (FP_FN_error + 2 * TP + 1e-7)).cpu().numpy()
        # Pixel Accuracy
        metrics["Pixel Accuracy"] = (torch.sum(intersection) / torch.sum(union)).item()

        return metrics

    def __save_model(self, model, output_dir, model_name, model_type):
        """
        Save the model to the output directory with specific naming.

        Parameters:
            model (torch.nn.Module): Trained model to save
            output_dir (str): Directory where the model should be saved
            model_name (str): Name of the model
            model_type (str): Type of the model ("best" or "last")
        """
        os.makedirs(output_dir, exist_ok=True)  # Ensure the models directory exists
        model_path = os.path.join(output_dir, f"{model_name}_{model_type}.pth")
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model ({model_type}) saved as {model_path}.")
        return model_path

    def train_model_with_mlflow(self,output_dir):
        mlflow.set_tracking_uri(r"../mlruns")
        mlflow.set_experiment(f"Satellite Segmentation of Nepal - {self.output_model_name}")
        mlflow.start_run()  # Start the MLflow run

        # Log model parameters
        mlflow.log_param("epochs", self.num_epochs)
        mlflow.log_param("learning_rate", self.optimizer.param_groups[0]["lr"])
        mlflow.log_param("batch_size", self.train_data_generator.batch_size)

        best_val_loss = float("inf")  # Track the best validation loss
        models_dir = os.path.join(output_dir, "models")  # Directory to save models
        last_epoch_model_path = None  # Path for the last epoch model

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_metrics = {"IoU": [], "F1-Score": [], "Pixel Accuracy": 0.0}

            for images, masks in self.train_data_generator:
                images = images.to(self.device)
                masks = masks.long().to(self.device)

                # Forward pass
                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # Calculate metrics
                batch_metrics = self.__calculate_metrics(outputs, masks)
                for key in train_metrics.keys():
                    if isinstance(train_metrics[key], list):
                        train_metrics[key].append(batch_metrics[key].mean())
                    else:
                        train_metrics[key] += batch_metrics[key]

            avg_train_loss = train_loss / len(self.train_data_generator)
            avg_train_metrics = {k: (sum(v) / len(v) if isinstance(v, list) else v / len(self.train_data_generator))
                                 for k, v in train_metrics.items()}
            mlflow.log_metrics(avg_train_metrics, step=epoch)  # Log training metrics
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Train Metrics: {avg_train_metrics}")

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_metrics = {"IoU": [], "F1-Score": [], "Pixel Accuracy": 0.0}

            with torch.no_grad():
                for images, masks in self.val_data_generator:
                    images = images.to(self.device)
                    masks = masks.long().to(self.device)

                    # Forward pass
                    outputs = self.model(images)["out"]
                    loss = self.criterion(outputs, masks)
                    val_loss += loss.item()

                    # Calculate metrics
                    batch_metrics = self.__calculate_metrics(preds=outputs, targets=masks)
                    for key in val_metrics.keys():
                        if isinstance(val_metrics[key], list):
                            val_metrics[key].append(batch_metrics[key].mean())
                        else:
                            val_metrics[key] += batch_metrics[key]

            avg_val_loss = val_loss / len(self.val_data_generator)
            avg_val_metrics = {k: (sum(v) / len(v) if isinstance(v, list) else v / len(self.val_data_generator))
                               for k, v in val_metrics.items()}
            mlflow.log_metrics(avg_val_metrics, step=epoch)  # Log validation metrics

            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val Metrics: {avg_val_metrics}")

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            for key in avg_train_metrics.keys():
                mlflow.log_metric(f"train_{key}", avg_train_metrics[key], step=epoch)
                mlflow.log_metric(f"val_{key}", avg_val_metrics[key], step=epoch)

            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = self.__save_model(self.model, models_dir, self.output_model_name, "best")
                mlflow.log_artifact(best_model_path)
                print(
                    f"Best model updated and saved as {best_model_path} with validation loss: {best_val_loss:.4f}")

            # Save the model from the current (last) epoch
            last_epoch_model_path = self.__save_model(self.model, models_dir, self.output_model_name, "last")

        # Save the final model to MLflow
        mlflow.pytorch.log_model(self.model, "model")
        mlflow.log_artifact(last_epoch_model_path)
        print(f"Final model from the last epoch saved as {last_epoch_model_path}")

        mlflow.end_run()
        return self.model