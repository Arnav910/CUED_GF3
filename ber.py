from pathlib import Path
import argparse


def count_bit_errors(reference: bytes, recovered: bytes) -> tuple[int, int, int]:
    """
    Return bit errors, reference bit count, and compared byte count.

    If the recovered file is shorter or longer than the reference, every missing
    or extra byte is counted as 8 bit errors.
    """
    compared_bytes = min(len(reference), len(recovered))
    bit_errors = 0

    for ref_byte, rec_byte in zip(reference[:compared_bytes], recovered[:compared_bytes]):
        bit_errors += (ref_byte ^ rec_byte).bit_count()

    length_difference = abs(len(reference) - len(recovered))
    bit_errors += length_difference * 8

    reference_bits = len(reference) * 8
    return bit_errors, reference_bits, compared_bytes


def calculate_ber(reference_path: Path, recovered_path: Path) -> float:
    reference = reference_path.read_bytes()
    recovered = recovered_path.read_bytes()

    if len(reference) == 0:
        raise ValueError("Reference file is empty, so BER is undefined.")

    bit_errors, reference_bits, compared_bytes = count_bit_errors(reference, recovered)
    ber = bit_errors / reference_bits

    print(f"Reference file: {reference_path}")
    print(f"Recovered file: {recovered_path}")
    print(f"Reference bytes: {len(reference)}")
    print(f"Recovered bytes: {len(recovered)}")
    print(f"Compared bytes: {compared_bytes}")
    print(f"Bit errors: {bit_errors}")
    print(f"Total reference bits: {reference_bits}")
    print(f"BER: {ber:.12f}")

    return ber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare test.txt and recovered_file.bin and report bit error rate."
    )
    parser.add_argument(
        "reference",
        nargs="?",
        default="test.txt",
        help="Original file to compare against. Defaults to test.txt.",
    )
    parser.add_argument(
        "recovered",
        nargs="?",
        default="recovered_file.bin",
        help="Recovered file to measure. Defaults to recovered_file.bin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calculate_ber(Path(args.reference), Path(args.recovered))


if __name__ == "__main__":
    main()