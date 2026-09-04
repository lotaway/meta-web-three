<script lang="ts" setup>
import { onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { type FormInstance, type FormRules } from 'element-plus'
import { generateCaptchaAPI, sendVerificationCodeAPI, registerDeveloperAPI, type CaptchaChallenge } from '@/apis/developer-portal'

const router = useRouter()
const registerFormRef = ref<FormInstance>()
const loading = ref(false)
const sendingCode = ref(false)
const countdown = ref(0)
const captcha = ref<CaptchaChallenge | null>(null)

const registerForm = reactive({
  name: '',
  email: '',
  phone: '',
  description: '',
  captchaToken: '',
  captchaAnswer: '',
  emailCode: '',
})

const emailRule = { required: true, trigger: 'blur', message: '请输入公司邮箱' }
const registerRules = reactive<FormRules<typeof registerForm>>({
  name: [
    { required: true, trigger: 'blur', message: '请输入姓名' },
    { min: 2, max: 128, trigger: 'blur', message: '姓名长度在 2 到 128 个字符之间' },
  ],
  email: [
    emailRule,
    { type: 'email', trigger: 'blur', message: '邮箱格式不正确' },
  ],
  phone: [
    { pattern: /^[0-9+\-\s]{6,32}$/, trigger: 'blur', message: '手机号格式不正确' },
  ],
  description: [
    { max: 1000, trigger: 'blur', message: '简介不能超过 1000 个字符' },
  ],
  captchaAnswer: [
    { required: true, trigger: 'blur', message: '请输入图形验证码' },
    { max: 8, trigger: 'blur', message: '验证码不能超过 8 个字符' },
  ],
  emailCode: [
    { required: true, trigger: 'blur', message: '请输入邮箱验证码' },
    { len: 6, trigger: 'blur', message: '邮箱验证码为 6 位数字' },
  ],
})

onMounted(() => {
  refreshCaptcha()
})

async function refreshCaptcha() {
  try {
    const res = await generateCaptchaAPI()
    captcha.value = res.data
    registerForm.captchaToken = res.data.token
    registerForm.captchaAnswer = ''
  } catch (e) {
    // 拦截器已提示错误
    console.error(e)
  }
}

async function sendVerificationCode() {
  if (!registerForm.email) {
    ElMessage.warning('请先填写邮箱')
    return
  }
  sendingCode.value = true
  try {
    await sendVerificationCodeAPI(registerForm.email.trim())
    ElMessage.success('验证码已发送，请查收邮箱')
    countdown.value = 60
    const timer = window.setInterval(() => {
      countdown.value -= 1
      if (countdown.value <= 0) {
        window.clearInterval(timer)
      }
    }, 1000)
  } catch (e) {
    console.error(e)
  } finally {
    sendingCode.value = false
  }
}

async function handleRegister() {
  registerFormRef.value!.validate(async (valid) => {
    if (!valid) {
      return
    }
    loading.value = true
    try {
      await registerDeveloperAPI({
        name: registerForm.name.trim(),
        email: registerForm.email.trim(),
        phone: registerForm.phone.trim(),
        description: registerForm.description.trim(),
        captchaToken: registerForm.captchaToken,
        captchaAnswer: registerForm.captchaAnswer.trim().toUpperCase(),
        emailCode: registerForm.emailCode.trim(),
      })
      ElMessage.success('注册申请已提交，请等待管理员审核')
      router.push({ path: '/login' })
    } catch (e) {
      refreshCaptcha()
      console.error(e)
    } finally {
      loading.value = false
    }
  })
}
</script>

<template>
  <div>
    <el-card class="register-form-layout">
      <template #header>
        <div class="register-title">开发者注册</div>
      </template>
      <el-form :model="registerForm" :rules="registerRules" ref="registerFormRef" label-position="left">
        <el-form-item prop="name">
          <el-input name="name" v-model="registerForm.name" placeholder="姓名 / 联系人" />
        </el-form-item>
        <el-form-item prop="email">
          <el-input name="email" v-model="registerForm.email" placeholder="公司邮箱" />
        </el-form-item>
        <el-form-item prop="phone">
          <el-input name="phone" v-model="registerForm.phone" placeholder="联系电话（选填）" />
        </el-form-item>
        <el-form-item prop="description">
          <el-input name="description" type="textarea" :rows="2" v-model="registerForm.description" placeholder="开发者简介（选填）" />
        </el-form-item>
        <el-form-item prop="captchaAnswer">
          <el-input name="captchaAnswer" v-model="registerForm.captchaAnswer" placeholder="图形验证码" @keyup.enter="handleRegister">
            <template #append>
              <img
                v-if="captcha"
                class="captcha-image"
                :src="captcha.image"
                alt="captcha"
                title="点击刷新"
                @click="refreshCaptcha"
              />
            </template>
          </el-input>
        </el-form-item>
        <el-form-item prop="emailCode">
          <el-input name="emailCode" v-model="registerForm.emailCode" placeholder="邮箱验证码" @keyup.enter="handleRegister">
            <template #append>
              <el-button :loading="sendingCode" :disabled="countdown > 0" @click="sendVerificationCode">
                {{ countdown > 0 ? `${countdown}s` : '获取验证码' }}
              </el-button>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-button style="width: 45%" type="primary" :loading="loading" @click="handleRegister">
            提交注册
          </el-button>
          <el-button style="width: 30%" @click="router.push({ path: '/login' })">返回登录</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<style scoped>
.register-form-layout {
  position: absolute;
  left: 0;
  right: 0;
  width: 420px;
  margin: 80px auto;
  border-top: 10px solid #409EFF;
}
.register-title {
  text-align: center;
  font-size: 16px;
  font-weight: bold;
}
.captcha-image {
  display: block;
  height: 28px;
  cursor: pointer;
}
</style>