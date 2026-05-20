package com.appsdk

import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.module.annotations.ReactModule

import androidx.credentials.CredentialManager
import androidx.credentials.CreatePublicKeyCredentialRequest
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetPublicKeyCredentialOption
import androidx.credentials.exceptions.CreateCredentialException
import androidx.credentials.exceptions.GetCredentialException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

import java.math.BigDecimal
import java.security.MessageDigest
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

@ReactModule(name = AppsdkModule.NAME)
class AppsdkModule(reactContext: ReactApplicationContext) : NativeAppsdkSpec(reactContext) {

  private val mainScope = CoroutineScope(Dispatchers.Main)
  private val credentialManager by lazy { CredentialManager.create(reactContext) }

  override fun getName(): String = NAME

  override fun generateRequestSignature(params: ReadableMap, secretKey: String): String {
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
    return digest.joinToString("") { "%02x".format(it) }
  }

  override fun preciseAmountSum(amountA: String, amountB: String): String {
    return try {
        BigDecimal(amountA).add(BigDecimal(amountB)).toPlainString()
    } catch (e: Exception) {
        "0"
    }
  }

  override fun computeOrderTotal(unitPrice: String, quantity: Double, discountAmount: String, shippingFee: String): String {
    return try {
        val price = BigDecimal(unitPrice)
        val subtotal = price.multiply(BigDecimal(quantity))
        val total = subtotal.subtract(BigDecimal(discountAmount)).add(BigDecimal(shippingFee))
        total.toPlainString()
    } catch (e: Exception) {
        "0"
    }
  }

  override fun hmacSign(message: String, signingKey: String): String {
    val mac = Mac.getInstance("HmacSHA256")
    val secretKeySpec = SecretKeySpec(signingKey.toByteArray(), "HmacSHA256")
    mac.init(secretKeySpec)
    val hmac = mac.doFinal(message.toByteArray())
    return hmac.joinToString("") { "%02x".format(it) }
  }

  override fun createNonce(): String {
    return UUID.randomUUID().toString().replace("-", "")
  }

  override fun systemTimestampMs(): Double {
    return System.currentTimeMillis().toDouble()
  }

  override fun createPasskey(rpId: String, userName: String, promise: Promise) {
    val activity = reactApplicationContext.getCurrentActivity()
    if (activity == null) {
      promise.reject("ACTIVITY_NOT_FOUND", "Cannot find current activity")
      return
    }

    val challenge = UUID.randomUUID().toString().toByteArray()
    val userId = UUID.randomUUID().toString().toByteArray()

    // 使用 android.util.Base64 以支持 API 24
    val challengeBase64 = Base64.encodeToString(challenge, Base64.URL_SAFE or Base64.NO_PADDING or Base64.NO_WRAP)
    val userIdBase64 = Base64.encodeToString(userId, Base64.URL_SAFE or Base64.NO_PADDING or Base64.NO_WRAP)

    val requestJson = """
      {
        "challenge": "$challengeBase64",
        "rp": { "name": "Meta Web Three", "id": "$rpId" },
        "user": {
            "id": "$userIdBase64",
            "name": "$userName",
            "displayName": "$userName"
        },
        "pubKeyCredParams": [{ "type": "public-key", "alg": -7 }],
        "authenticatorSelection": {
            "authenticatorAttachment": "platform",
            "residentKey": "required",
            "userVerification": "required"
        },
        "timeout": 60000,
        "attestation": "none"
      }
    """.trimIndent()

    val request = CreatePublicKeyCredentialRequest(requestJson)

    mainScope.launch {
      try {
        val result = credentialManager.createCredential(activity, request)
        val response = result.data.getString("androidx.credentials.BUNDLE_KEY_REGISTRATION_RESPONSE_JSON")
        promise.resolve(response)
      } catch (e: CreateCredentialException) {
        promise.reject("PASSKEY_CREATE_ERROR", e.errorMessage?.toString() ?: "Unknown error")
      } catch (e: Exception) {
        promise.reject("UNKNOWN_ERROR", e.message)
      }
    }
  }

  override fun getPasskeyList(): WritableArray {
    // @TODO 隐私受限，返回空数组。真实场景需结合业务库
    return Arguments.createArray()
  }

  override fun authenticatePasskey(rpId: String, challenge: String, promise: Promise) {
    val activity = reactApplicationContext.getCurrentActivity()
    if (activity == null) {
      promise.reject("ACTIVITY_NOT_FOUND", "Cannot find current activity")
      return
    }

    val requestJson = """
      {
        "challenge": "$challenge",
        "rpId": "$rpId",
        "userVerification": "required"
      }
    """.trimIndent()

    val getPublicKeyCredentialOption = GetPublicKeyCredentialOption(requestJson)
    val getCredentialRequest = GetCredentialRequest(listOf(getPublicKeyCredentialOption))

    mainScope.launch {
      try {
        val result = credentialManager.getCredential(activity, getCredentialRequest)
        val response = result.credential.data.getString("androidx.credentials.BUNDLE_KEY_AUTHENTICATION_RESPONSE_JSON")
        promise.resolve(response)
      } catch (e: GetCredentialException) {
        promise.reject("PASSKEY_AUTH_ERROR", e.errorMessage?.toString() ?: "Authentication failed")
      } catch (e: Exception) {
        promise.reject("UNKNOWN_ERROR", e.message)
      }
    }
  }

  override fun deletePasskey(credentialId: String): Boolean {
    // @TODO Android 原生暂不提供删除接口
    return false
  }

  companion object {
    const val NAME = "Appsdk"
  }
}
