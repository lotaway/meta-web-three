<script setup lang=\"ts\">
import { ref, reactive } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Setting, Check, Close } from '@element-plus/icons-vue'

const { t } = useI18n()

const loading = ref(false)

const paymentConfig = reactive({
  wechat: {
    enabled: true,
    appId: '',
    mchId: '',
    apiKey: '',
    certPath: ''
  },
  alipay: {
    enabled: true,
    appId: '',
    privateKey: '',
    publicKey: '',
    gateway: 'https://openapi.alipay.com/gateway.do'
  },
  stripe: {
    enabled: false,
    publishableKey: '',
    secretKey: '',
    webhookSecret: ''
  },
  crypto: {
    enabled: false,
    network: 'ETH',
    contractAddress: '',
    rpcUrl: ''
  }
})

const handleSave = async (method: string) => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    ElMessage.success(t('payment.config.save.success') + ': ' + method)
  } catch (error) {
    ElMessage.error(t('payment.config.save.failed'))
  } finally {
    loading.value = false
  }
}

const handleTest = async (method: string) => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    ElMessage.success(t('payment.config.test.success') + ': ' + method)
  } catch (error) {
    ElMessage.error(t('payment.config.test.failed'))
  } finally {
    loading.value = false
  }
}

const paymentMethods = [
  {
    key: 'wechat',
    label: 'WeChat Pay',
    icon: 'Wallet',
    fields: ['appId', 'mchId', 'apiKey', 'certPath']
  },
  {
    key: 'alipay',
    label: 'Alipay',
    icon: 'Wallet',
    fields: ['appId', 'privateKey', 'publicKey', 'gateway']
  },
  {
    key: 'stripe',
    label: 'Stripe',
    icon: 'CreditCard',
    fields: ['publishableKey', 'secretKey', 'webhookSecret']
  },
  {
    key: 'crypto',
    label: 'Crypto Payment',
    icon: 'Coin',
    fields: ['network', 'contractAddress', 'rpcUrl']
  }
]
</script>

<template>
  <div class=\"app-container\">
    <el-card shadow=\"never\">
      <template #header>
        <div class=\"card-header\">
          <span>{{ t('payment.config.title') }}</span>
        </div>
      </template>

      <el-tabs v-model=\"activeTab\" type=\"border-card\">
        <el-tab-pane
          v-for=\"method in paymentMethods\"
          :key=\"method.key\"
          :label=\"method.label\"
          :name=\"method.key\"
        >
          <el-form :model=\"paymentConfig[method.key]\" label-width=\"150px\">
            <el-form-item :label=\"t('payment.config.enabled')\">
              <el-switch
                v-model=\"paymentConfig[method.key].enabled\"
                :active-text=\"t('common.enable')\"
                :inactive-text=\"t('common.disable')\"
              />
            </el-form-item>

            <el-divider />

            <el-form-item
              v-for=\"field in method.fields\"
              :key=\"field\"
              :label=\"t(`payment.config.fields.${field}`)\"
            >
              <el-input
                v-model=\"paymentConfig[method.key][field]\"
                :type=\"field.includes('Key') || field.includes('Key') || field === 'privateKey' ? 'password' : 'text'\"
                :placeholder=\"t(`payment.config.fields.${field}Placeholder`)\"
              />
            </el-form-item>

            <el-form-item>
              <el-button
                type=\"primary\"
                :icon=\"Check\"
                :loading=\"loading\"
                @click=\"handleSave(method.label)\"
              >
                {{ t('common.save') }}
              </el-button>
              <el-button
                :icon=\"Setting\"
                :loading=\"loading\"
                @click=\"handleTest(method.label)\"
              >
                {{ t('payment.config.test.button') }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-card shadow=\"never\" style=\"margin-top: 20px\">
      <template #header>
        <span>{{ t('payment.config.advanced.title') }}</span>
      </template>

      <el-form label-width=\"180px\">
        <el-form-item :label=\"t('payment.config.advanced.timeout')\">
          <el-input-number v-model=\"config.timeout\" :min=\"30\" :max=\"300\" />
          <span style=\"margin-left: 10px\">seconds</span>
        </el-form-item>

        <el-form-item :label=\"t('payment.config.advanced.retryCount')\">
          <el-input-number v-model=\"config.retryCount\" :min=\"0\" :max=\"5\" />
        </el-form-item>

        <el-form-item :label=\"t('payment.config.advanced.autoRefund')\">
          <el-switch
            v-model=\"config.autoRefund\"
            :active-text=\"t('common.enable')\"
            :inactive-text=\"t('common.disable')\"
          />
        </el-form-item>

        <el-form-item :label=\"t('payment.config.advanced.feeRate')\">
          <el-input-number v-model=\"config.feeRate\" :min=\"0\" :max=\"10\" :precision=\"2\" :step=\"0.1\" />
          <span style=\"margin-left: 10px\">%</span>
        </el-form-item>

        <el-form-item>
          <el-button type=\"primary\" @click=\"handleSave('Advanced')\">
            {{ t('common.save') }}
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script lang=\"ts\">
import { reactive, ref } from 'vue'

const config = reactive({
  timeout: 60,
  retryCount: 3,
  autoRefund: false,
  feeRate: 0.6
})

const activeTab = ref('wechat')

export default {
  setup() {
    return {
      config,
      activeTab
    }
  }
}
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.el-divider {
  margin: 20px 0;
}
</style>