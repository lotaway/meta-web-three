package com.appsdk

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.module.annotations.ReactModule
import java.math.BigDecimal
import java.security.MessageDigest
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

@ReactModule(name = AppsdkModule.NAME)
class AppsdkModule(reactContext: ReactApplicationContext) : NativeAppsdkSpec(reactContext) {

  override fun getName(): String = NAME

  override fun generateRequestSignature(params: ReadableMap, secretKey: String, promise: Promise) {
    val keyIterator = params.keySetIterator()
    val paramPairs = mutableListOf<String>()
    
    while (keyIterator.hasNextKey()) {
      val key = keyIterator.nextKey()
      val value = params.getString(key) ?: ""
      paramPairs.add("$key=$value")
    }
    
    paramPairs.sort()
    val message = paramPairs.joinToString("&") + secretKey
    
    val md = MessageDigest.getInstance("SHA-256")
    val digest = md.digest(message.toByteArray())
    val signature = digest.joinToString("") { "%02x".format(it) }
    
    promise.resolve(signature)
  }

  override fun preciseAmountSum(amountA: String, amountB: String, promise: Promise) {
    val sum = BigDecimal(amountA).add(BigDecimal(amountB))
    promise.resolve(sum.toPlainString())
  }

  override fun computeOrderTotal(unitPrice: String, quantity: Int, discountAmount: String, shippingFee: String, promise: Promise) {
    val price = BigDecimal(unitPrice)
    val subtotal = price.multiply(BigDecimal(quantity))
    val total = subtotal.subtract(BigDecimal(discountAmount)).add(BigDecimal(shippingFee))
    promise.resolve(total.toPlainString())
  }

  override fun hmacSign(message: String, signingKey: String, promise: Promise) {
    val mac = Mac.getInstance("HmacSHA256")
    val secretKeySpec = SecretKeySpec(signingKey.toByteArray(), "HmacSHA256")
    mac.init(secretKeySpec)
    val hmac = mac.doFinal(message.toByteArray())
    val result = hmac.joinToString("") { "%02x".format(it) }
    promise.resolve(result)
  }

  override fun createNonce(promise: Promise) {
    val nonce = UUID.randomUUID().toString().replace("-", "")
    promise.resolve(nonce)
  }

  override fun systemTimestampMs(promise: Promise) {
    promise.resolve(System.currentTimeMillis())
  }

  override fun createPasskey(rpId: String, userName: String, promise: Promise) {
    promise.resolve("passkey-${UUID.randomUUID()}")
  }

  override fun getPasskeyList(promise: Promise) {
    val list = WritableArray.createArray()
    list.pushString("passkey-1")
    list.pushString("passkey-2")
    promise.resolve(list)
  }

  override fun authenticatePasskey(challenge: String, promise: Promise) {
    promise.resolve(true)
  }

  override fun deletePasskey(credentialId: String, promise: Promise) {
    promise.resolve(null)
  }

  companion object {
    const val NAME = "Appsdk"
  }
}

