pragma circom 2.0.0;

template SignatureVerification() {
    signal private input privateKey
    signal private input messageHash
    signal output publicKeyX
    signal output publicKeyY

    var curveParams = EdDSA()
    var pubKey = curveParams.mulBase(privateKey)
    publicKeyX <== pubKey[0]
    publicKeyY <== pubKey[1]

    var sig = curveParams.sign(privateKey, messageHash)
    var isValid = curveParams.verify(pubKey, messageHash, sig)

    isValid === 1
}
component main = SignatureVerification()