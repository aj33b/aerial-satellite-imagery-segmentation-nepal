from sources.helpers.train import train_model_with_pan
from sources.helpers.utils import prepare_dataset, test_data_generator


def main():
    print("\nSATELLITE IMAGE LANDCOVER SEGMENTATION TOOL\n")
    print("Developed By: Ajeeb Rimal | M. Tech. AI 2021 | Kathmandu University\n")
    while True:
        print("\nPlease select an option:")
        print("1. Prepare dataset")
        print("2. Define & test data_generator")
        print("3. Train model")
        print("4. Evaluate model")
        print("5. Generate predictions")
        print("6. Exit")
        choice = input("Enter your choice [1-6] : ")

        # Prepare Dataset
        if choice == '1':
            print("Preparing dataset...")
            prepare_dataset()
            print("Dataset prepared!")

        # Define and test Data Generator
        elif choice == '2':
            print("Defining & testing data generator...")
            test_data_generator()

        # Train Model
        elif choice == '3':
            print("Training model with PAN...")
            train_model_with_pan()

        # Evaluate Model
        elif choice == '4':
            print("Evaluating model...")
            # evaluate_model()

        # Generate Predictions
        elif choice == '5':
            print("Generating predictions...")
            # generate_predictions()

        # Exit
        elif choice == '6':
            print("Exiting...")
            break

        else:
            print("Invalid choice! Please try again.")
            continue


if __name__ == "__main__":
    main()
