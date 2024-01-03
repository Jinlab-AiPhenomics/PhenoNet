from arch.phenonet import PhenoNet

def main():
    model = PhenoNet()
    print(model.print_architecture())

if __name__ == "__main__":
    main()