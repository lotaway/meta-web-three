package com.metawebthree.utils;

import org.web3j.crypto.*;
import org.web3j.utils.Numeric;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.SignatureException;
import java.util.Arrays;

public class Eip191Recover {

    public static String recoverAddressFromMessage(String message, String signatureHex) throws SignatureException {
        byte[] sigBytes = Numeric.hexStringToByteArray(signatureHex);
        Sign.SignatureData sig = toSignatureData(sigBytes);
        BigInteger publicKey = Sign.signedMessageToKey(
                message.getBytes(StandardCharsets.UTF_8),
                sig);
        return "0x" + Keys.getAddress(publicKey);
    }

    public static String recoverAddressFromHash32(byte[] messageHash32, String signatureHex) {
        if (messageHash32.length != 32) {
            throw new IllegalArgumentException("messageHash32 must be 32 bytes");
        }

        byte[] sigBytes = Numeric.hexStringToByteArray(signatureHex);
        Sign.SignatureData sig = toSignatureData(sigBytes);

        byte[] prefix = "\u0019Ethereum Signed Message:\n32".getBytes(StandardCharsets.UTF_8);
        byte[] ethSignedHash = Hash.sha3(concat(prefix, messageHash32));

        int recId = (sig.getV()[0] & 0xFF) - 27;
        ECDSASignature ecdsa = new ECDSASignature(
                new BigInteger(1, sig.getR()),
                new BigInteger(1, sig.getS()));

        BigInteger publicKey = Sign.recoverFromSignature(recId, ecdsa, ethSignedHash);
        if (publicKey == null) {
            throw new IllegalStateException("Could not recover public key");
        }
        return "0x" + Keys.getAddress(publicKey);
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

    private static byte[] concat(byte[] a, byte[] b) {
        byte[] out = new byte[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
