import os
import io
import struct

# The bits of resolution we use to represent probabilities
# Finest representation is 1 / 2**X, so with 12 => 0.024%.
# Another way of looking at this is you have 2 ** 12 => 4096
# "tokens" to distribute over all of the bytes values, since
# this is a byte-oritented codec.
MODEL_BITS = 12

# This is should be set to 1 bit less than whatever integer size you
# want to commonly operate on. In the 64-bit Rust implementation, this
# is set to 63.
STATE_BITS = 31

# This is what makes it a byte-oriented codec. It will always write and
# read in chunks ofthis many bits.
BITS_PER_CHUNK = 8


def generate_frequency_table(raw):
    enc_total = 1 << MODEL_BITS
    freqs = [0] * 256  # There are 256 discrete values of a byte
    for b in raw:
        freqs[b] += 1
    total = sum(freqs)

    # TODO: better explanation
    # Sidestep the scaled freq less than 2 issue. Alternatively, see Ryg's solution:
    # https://github.com/rygorous/ryg_rans/blob/master/rans_byte.h#L199-L229
    # It feels like this is an iterative solution to the stealing method he implements,
    # but I haven't been through both in enough detail yet.
    while True:
        added = 0
        limit = (total // enc_total) + 1
        for i, f in enumerate(freqs):
            if f != 0 and f < limit:
                added += limit - f
                freqs[i] = limit

        if added == 0:
            break

        total += added

    freqs = [f * enc_total // total for f in freqs]
    return freqs


class StaticModel:
    ID = 1

    def __init__(self, input_data):
        self._freqs = generate_frequency_table(input_data)

        self._cdf = []
        cf = 0
        for f in self._freqs:
            self._cdf.append(cf)
            cf += f

        self._mask = (1 << MODEL_BITS) - 1

    def cdf(self, symbol):
        return self._cdf[symbol]

    def freq(self, symbol):
        return self._freqs[symbol]


def compress(model, raw):
    # You'll notice we're NOT reversing the input or output streams in the encoder. This means
    # that the decoder will have to do this reversal. Why do it this way? Why does everyone else
    # push work to the encoder?
    #
    # 1. Most use-cases are write once, read many. It makes sense to have the read-many case be fast/cheap.
    # 2. I have elected to use a minimal-memory approach for the encoder. The Rust implementation has
    #    tradeoffs here, but I think this is an interesting case for when the encoder needs to maintain
    #    a large number of simultaneous but separate streams.

    # TODO: Is the value 8 here arbitrary? What are the limits? Need to reread the paper.
    L = 1 << (STATE_BITS - 8)
    X = L
    MASK = (1 << MODEL_BITS) - 1
    outbuf = []

    for b in raw:
        f = model.freq(b)
        if f == 0:
            raise ValueError("Cannot encode symbols with zero frequency")

        # This computes a max value that if our state value is over, it will
        # overflow our state bits if we do the update step. Not an issue in Python
        # since it has arbitrary precision arithmetic, but a major problem otherwise.
        H = ((L >> MODEL_BITS) << 8) * f
        while X >= H:
            outbuf.append(X & 0xFF)
            X >>= 8
        # Straight from wikipedia / Jarek's paper (wiki's notation):
        # C(x,s) = (floor(x / f[s]) << n) + (x % f[s]) + CDF[s]
        X = ((X // f) << MODEL_BITS) + (X % f) + model.cdf(b)

    # Hard-code for state_bits = 31
    # We want the encoder to produce backward symbols. NOTE:
    # This is opposite of many others (Including Ryg's) where
    # all fungible cost is pushed to the encoder. In my implementation,
    # we will want both but for now chose lowest possible encoder memory.
    # (See above description)
    return bytes(outbuf) + struct.pack("<I", X)


def decompress(model, comp, length):
    # Require that the first 4 bytes exist so we can correctly initialize our state
    comp = io.BytesIO(bytes(reversed(comp)))
    outbuf = []

    # Encode as little-endian, decode as big (the decompressor does the reversing)
    X = struct.unpack(">I", comp.read(4))[0]
    MASK = (1 << MODEL_BITS) - 1

    # Build the symbol table that lets us find the encoded symbol for each of the
    # 2 ** MODEL_BITS values that we could have encoded.
    symbol_table = [0] * (MASK + 1)
    cursor = 0
    for s in range(256):
        up = model.cdf(s)
        while cursor < up:
            symbol_table[cursor] = s - 1
            cursor += 1

    # TODO: Again, what is up with the value 8 here?
    L = 1 << (STATE_BITS - 8)

    for _ in range(length):
        s = symbol_table[X & MASK]
        if model.freq(s) == 0:
            raise ValueError(
                "Decoding/Symbol Table Error, symbol with 0 frequency found."
            )

        X = model.freq(s) * (X >> MODEL_BITS) + (X & MASK) - model.cdf(s)
        while X < L:
            X = (X << 8) | comp.read(1)[0]
        outbuf.append(s)

    return bytes(reversed(outbuf))


def main():
    from argparse import ArgumentParser

    p = ArgumentParser()

    p.add_argument("input_file")
    args = p.parse_args()

    data = open(args.input_file, "rb").read()
    print("Read {} bytes, compressing...".format(len(data)))

    model = StaticModel(data)
    compressed = compress(model, data)
    decompressed = decompress(model, compressed, len(data))

    print("Compression round trip matches: {}".format(data == decompressed))
    print(
        "Ratio 1:{:.3f}, compressed is {:5.2f}% of original".format(
            len(data) / len(compressed), 100.0 * len(compressed) / len(data)
        )
    )


if __name__ == "__main__":
    main()
