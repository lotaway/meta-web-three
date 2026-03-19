package com.metawebthree.utils;

import org.web3j.crypto.ECDSASignature;
import org.web3j.crypto.Hash;
import org.web3j.crypto.Keys;
import org.web3j.crypto.Sign;
import org.web3j.utils.Numeric;

import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.SignatureException;
import java.sql.Timestamp;
import java.util.Arrays;

public class Eip712Recover {

    public static final String DEFAULT_TYPE_DECLARATION =
            "Eip712Data(address address,uint256 amount,uint256 ts,string other)";

    public static String recoverAddress(byte[] domainSeparator,
                                        byte[] address,
                                        BigDecimal amount,
                                        Timestamp ts,
                                        String other,
                                        String signatureHex) throws SignatureException {
        byte[] messageHash32 = hashTypedData(domainSeparator, address, amount, ts, other);
        return recoverAddressFromHash32(messageHash32, signatureHex);
    }

    public static String recoverAddressFromHash32(byte[] messageHash32, String signatureHex)
            throws SignatureException {
        if (messageHash32.length != 32) {
            throw new IllegalArgumentException("messageHash32 must be 32 bytes");
        }

        byte[] sigBytes = Numeric.hexStringToByteArray(signatureHex);
        Sign.SignatureData sig = toSignatureData(sigBytes);

        int recId = (sig.getV()[0] & 0xFF) - 27;
        ECDSASignature ecdsa = new ECDSASignature(
                new BigInteger(1, sig.getR()),
                new BigInteger(1, sig.getS()));

        BigInteger publicKey = Sign.recoverFromSignature(recId, ecdsa, messageHash32);
        if (publicKey == null) {
            throw new IllegalStateException("Could not recover public key");
        }
        return "0x" + Keys.getAddress(publicKey);
    }

    public static byte[] hashTypedData(byte[] domainSeparator,
                                       byte[] address,
                                       BigDecimal amount,
                                       Timestamp ts,
                                       String other) {
        if (domainSeparator.length != 32) {
            throw new IllegalArgumentException("domainSeparator must be 32 bytes");
        }

        byte[] structHash = hashStruct(DEFAULT_TYPE_DECLARATION, address, amount, ts, other);
        byte[] prefix = new byte[]{0x19, 0x01};
        return Hash.sha3(concat(prefix, domainSeparator, structHash));
    }

    public static byte[] hashStruct(String typeDeclaration,
                                    byte[] address,
                                    BigDecimal amount,
                                    Timestamp ts,
                                    String other) {
        byte[] typeHash = Hash.sha3(typeDeclaration.getBytes(StandardCharsets.UTF_8));
        byte[] addressWord = addressWord(address);
        byte[] amountWord = uint256Word(toBigIntegerExact(amount));
        byte[] tsWord = uint256Word(BigInteger.valueOf(ts.toInstant().getEpochSecond()));
        byte[] otherHash = Hash.sha3(other == null
                ? new byte[0]
                : other.getBytes(StandardCharsets.UTF_8));

        return Hash.sha3(concat(typeHash, addressWord, amountWord, tsWord, otherHash));
    }

    private static BigInteger toBigIntegerExact(BigDecimal value) {
        if (value == null) {
            throw new IllegalArgumentException("amount must not be null");
        }
        return value.stripTrailingZeros().toBigIntegerExact();
    }

    private static byte[] uint256Word(BigInteger value) {
        if (value.signum() < 0) {
            throw new IllegalArgumentException("uint256 must be >= 0");
        }
        byte[] raw = Numeric.toBytesPadded(value, 32);
        return raw;
    }

    private static byte[] addressWord(byte[] address) {
        if (address == null || address.length == 0) {
            return Numeric.toBytesPadded(BigInteger.ZERO, 32);
        }
        if (address.length != 20) {
            throw new IllegalArgumentException("address must be 20 bytes");
        }
        return Numeric.toBytesPadded(new BigInteger(1, address), 32);
    }

    private static Sign.SignatureData toSignatureData(byte[] signatureBytes) {
        if (signatureBytes.length != 65) {
            throw new IllegalArgumentException("signature must be 65 bytes");
        }
        byte v = signatureBytes[64];
        if (v < 27) {
            v += 27;
        }
        byte[] r = Arrays.copyOfRange(signatureBytes, 0, 32);
        byte[] s = Arrays.copyOfRange(signatureBytes, 32, 64);
        return new Sign.SignatureData(v, r, s);
    }

    private static byte[] concat(byte[]... arrays) {
        int total = 0;
        for (byte[] array : arrays) {
            total += array.length;
        }
        ByteBuffer buffer = ByteBuffer.allocate(total);
        for (byte[] array : arrays) {
            buffer.put(array);
        }
        return buffer.array();
    }
}
