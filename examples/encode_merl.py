from brdf_enc_dec.model import BRDFNPs


def main():
    model = BRDFNPs(device="cpu", checkpoint="/app/model_weights/encdec/600k.pth")


if __name__ == "__main__":
    main()
