def main():
    print("=================================================================")
    print(" Welcome to the Anomaly Detection in Power Consumption Project!  ")
    print("=================================================================")
    print("\nProject Structure and Navigation Commands:\n")
    print("1. To generate data and train all autoencoder models:")
    print("   python src/train.py\n")
    print("2. To evaluate models and view anomaly metrics:")
    print("   python src/evaluate.py\n")
    print("3. To launch the interactive web dashboard:")
    print("   streamlit run app/dashboard.py\n")
    print("Ensure you have installed all dependencies via:")
    print("   pip install -r requirements.txt")
    print("=================================================================")

if __name__ == "__main__":
    main()
